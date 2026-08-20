/**
 * collatz.cpp - Host code for the AMD HIP Collatz verifier
 *
 * Core algorithm: exact k-step Terras/Everett acceleration table.
 *
 * Key optimizations (see collatz.hip for kernel details):
 *   - Odd-only verification (even n=2m is covered by Collatz(m))
 *   - 8-byte packed jump table (add|mul|steps in one uint64)
 *   - k=18 table (2 MB) fully resident in L2 cache
 *   - Double-buffered async streams with GPU-side reduction
 *   - Checkpoint/resume support with versioned binary format
 *   - Counterexample detection with CPU re-verification
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <hip/hip_runtime.h>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// Checkpoint binary format identifiers.  Adding these prevents silent
// corruption when the struct layout changes in future versions.
#define CHECKPOINT_MAGIC   0x43545A4Cu  // "CTZL"
#define CHECKPOINT_VERSION 1u

struct Config {
    uint64_t batchSize = 1024ull * 1024 * 1024;   // 1G odd numbers/batch
    uint32_t maxSteps = 100000;
    int saveInterval = 10;
};

struct Checkpoint {
    uint32_t magic;
    uint32_t version;
    uint64_t lastNumber;
    uint64_t totalVerified;
    uint64_t totalSteps;
    uint32_t maxSteps;
    uint64_t maxStepsNumber;
    uint64_t batchCount;
    double totalTime;
    uint64_t lastVerifiedRange;
    uint64_t totalRealVerified;
};

// Return a checkpoint with safe default values.
static Checkpoint defaultCheckpoint() {
    Checkpoint cp;
    std::memset(&cp, 0, sizeof(cp));
    cp.magic = CHECKPOINT_MAGIC;
    cp.version = CHECKPOINT_VERSION;
    cp.lastNumber = 1;
    cp.maxStepsNumber = 1;
    return cp;
}

Checkpoint loadCheckpoint() {
    Checkpoint cp = defaultCheckpoint();
    std::ifstream file("checkpoint.bin", std::ios::binary);
    if (file.is_open()) {
        file.read((char*)&cp, sizeof(Checkpoint));
        // Reject truncated files or outdated/unknown formats.
        if (!file || cp.magic != CHECKPOINT_MAGIC || cp.version != CHECKPOINT_VERSION) {
            cp = defaultCheckpoint();
        }
        file.close();
    }
    return cp;
}

void saveCheckpoint(const Checkpoint& cp) {
    // Ensure magic and version are always correct on disk.
    Checkpoint tmp = cp;
    tmp.magic = CHECKPOINT_MAGIC;
    tmp.version = CHECKPOINT_VERSION;
    std::ofstream file("checkpoint.bin", std::ios::binary);
    if (file.is_open()) {
        file.write((char*)&tmp, sizeof(Checkpoint));
        file.close();
    }
}

std::string formatTime(double seconds) {
    if (seconds < 60) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.1fs", seconds);
        return std::string(buf);
    } else if (seconds < 3600) {
        int mins = (int)(seconds / 60);
        int secs = (int)(seconds) % 60;
        char buf[64];
        snprintf(buf, sizeof(buf), "%dm %ds", mins, secs);
        return std::string(buf);
    } else {
        int hours = (int)(seconds / 3600);
        int mins = (int)((int)(seconds) % 3600) / 60;
        char buf[64];
        snprintf(buf, sizeof(buf), "%dh %dm", hours, mins);
        return std::string(buf);
    }
}

// Used only for re-verifying a suspected counterexample on the CPU (rare).
inline uint32_t verifyNumberCPU(uint64_t n, uint32_t maxSteps) {
    unsigned __int128 x = (unsigned __int128)n;
    uint32_t steps = 0;
    while (x > 1 && steps < maxSteps) {
        if (x & 1) x = x * 3 + 1;
        else x = x >> 1;
        steps++;
    }
    return steps;
}

// ============================================================
// Build the exact k-step acceleration table.
// T(n) = n/2 (even) or (3n+1)/2 (odd). After TABLE_BITS applications:
//   T^k(n) = (mul(r)*n + add(r)) >> k,   r = n mod 2^k
// mul/add/steps depend only on r - proven by induction.
// Each entry is bit-packed into a uint64: add(29b)|mul(29b)|steps(6b).
// ============================================================
static const uint32_t TABLE_BITS = 18;
static const uint32_t TABLE_SIZE = 1u << TABLE_BITS;

// v9: bit-packed entry – add(29b)+mul(29b)+steps(6b)=64b
#define ADD_MASK   0x1FFFFFFFull
#define MUL_SHIFT  29
#define STEP_SHIFT 58

void buildJumpTable(uint64_t* table) {
    for (uint32_t r = 0; r < TABLE_SIZE; r++) {
        unsigned __int128 curMul = 1, curAdd = 0;
        uint64_t v = r;
        uint64_t pow2i = 1;
        uint32_t rawSteps = 0;
        for (uint32_t i = 0; i < TABLE_BITS; i++) {
            if (v & 1) {
                curMul = curMul * 3;
                curAdd = curAdd * 3 + pow2i;
                v = (3 * v + 1) >> 1;
                rawSteps += 2;
            } else {
                v = v >> 1;
                rawSteps += 1;
            }
            pow2i <<= 1;
        }
        table[r] = ((uint64_t)rawSteps << STEP_SHIFT)
                 | ((uint64_t)(uint32_t)curMul << MUL_SHIFT)
                 | (uint64_t)(uint32_t)curAdd;
    }
}

extern "C" {
    __global__ void collatzKernel(
        uint32_t maxStepsLimit, uint64_t numElements, uint64_t startValue,
        const uint64_t* jumpTable,
        uint64_t* outVerified, uint64_t* outTotalSteps,
        uint32_t* outMaxSteps, uint64_t* outMaxNumber,
        uint32_t* foundCounter
    );
}

struct GPUResources {
    uint64_t* d_outVerified[2];
    uint64_t* d_outTotalSteps[2];
    uint32_t* d_outMaxSteps[2];
    uint64_t* d_outMaxNumber[2];
    uint32_t* d_foundCounter[2];
    hipStream_t streams[2];

    uint64_t* d_jumpTable;
    uint64_t* h_jumpTable;
};

void cleanupGPU(GPUResources& res) {
    for (int i = 0; i < 2; i++) {
        if (res.d_outVerified[i]) hipFree(res.d_outVerified[i]);
        if (res.d_outTotalSteps[i]) hipFree(res.d_outTotalSteps[i]);
        if (res.d_outMaxSteps[i]) hipFree(res.d_outMaxSteps[i]);
        if (res.d_outMaxNumber[i]) hipFree(res.d_outMaxNumber[i]);
        if (res.d_foundCounter[i]) hipFree(res.d_foundCounter[i]);
        if (res.streams[i]) hipStreamDestroy(res.streams[i]);
    }
    if (res.d_jumpTable) hipFree(res.d_jumpTable);
}

void initGPU(GPUResources& res, Config& config) {
    for (int i = 0; i < 2; i++) {
        HIP_CHECK(hipMalloc(&res.d_outVerified[i], sizeof(uint64_t)));
        HIP_CHECK(hipMalloc(&res.d_outTotalSteps[i], sizeof(uint64_t)));
        HIP_CHECK(hipMalloc(&res.d_outMaxSteps[i], sizeof(uint32_t)));
        HIP_CHECK(hipMalloc(&res.d_outMaxNumber[i], sizeof(uint64_t)));
        HIP_CHECK(hipMalloc(&res.d_foundCounter[i], sizeof(uint32_t)));
        HIP_CHECK(hipStreamCreate(&res.streams[i]));
    }

    HIP_CHECK(hipMalloc(&res.d_jumpTable, TABLE_SIZE * sizeof(uint64_t)));
    HIP_CHECK(hipMemcpy(res.d_jumpTable, res.h_jumpTable, TABLE_SIZE * sizeof(uint64_t), hipMemcpyHostToDevice));
}

int main() {
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipDeviceSetLimit(hipLimitMallocHeapSize, 14ULL * 1024 * 1024 * 1024));

    printf("\033[2J\033[H");
    printf("============================================================\n");
    printf("  AMD HIP Collatz Verifier (corrected)\n");
    printf("  Exact k=%u bit acceleration table\n", TABLE_BITS);
    printf("============================================================\n\n");

    printf("[Info] Building exact k=%u acceleration table (%u entries)...\n", TABLE_BITS, TABLE_SIZE);
    fflush(stdout);
    uint64_t* h_jumpTable = new uint64_t[TABLE_SIZE];
    buildJumpTable(h_jumpTable);
    printf("[Info] All tables ready!\n\n");

    Config config;
    GPUResources res;
    res.h_jumpTable = h_jumpTable;

    initGPU(res, config);

    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    printf("[GPU] Found %d AMD GPU(s)\n", deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, i));
        printf("   GPU %d: %s\n", i, prop.name);
        printf("   Compute Units: %d\n", prop.multiProcessorCount);
        printf("   Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
        printf("   Warp size: %d\n", prop.warpSize);
    }
    printf("\n");

    printf("[Info] Batch size: %lluM odd numbers/batch (x2 buffers, covers %lluM total)\n",
           (unsigned long long)(config.batchSize / (1024*1024)),
           (unsigned long long)(config.batchSize * 2 / (1024*1024)));
    printf("[Info] Mode: odd-only verification (evens covered by n/2)\n");
    printf("[Info] Acceleration table: %u bits (~%.1f MB, 8B packed, 2x unroll)\n", TABLE_BITS,
           (double)(TABLE_SIZE * sizeof(uint64_t)) / (1024*1024));
    printf("\n");

    Checkpoint cp = loadCheckpoint();
    uint64_t currentNumber = cp.lastNumber;
    uint64_t totalVerified = cp.totalVerified;
    uint64_t totalSteps = cp.totalSteps;
    uint32_t maxSteps = cp.maxSteps;
    uint64_t maxStepsNumber = cp.maxStepsNumber;
    uint64_t batchCount = cp.batchCount;
    uint64_t totalRealVerified = cp.totalRealVerified;

    if (totalRealVerified < currentNumber - 1) totalRealVerified = currentNumber - 1;
    if (totalRealVerified == 0 && currentNumber > 1) totalRealVerified = currentNumber - 1;
    if (totalVerified > totalRealVerified) totalVerified = totalRealVerified;

    // Odd-only verification: ensure start is odd (evens are covered by n/2)
    if ((currentNumber & 1ull) == 0) currentNumber++;

    printf("[Info] Starting from: %llu\n", (unsigned long long)currentNumber);
    printf("============================================================\n\n");

    auto startTime = std::chrono::high_resolution_clock::now();
    auto lastDisplay = startTime;
    uint64_t lastVerified = totalRealVerified;
    bool foundCounterExample = false;
    uint64_t counterExampleNumber = 0;
    uint64_t startValue = currentNumber;
    int activeBuffer = 0;
    bool hasPending = false;

    // 1024 threads per block reduces block-scheduling overhead vs 512.
    const int threads = 1024;

    printf("Running...\n\n");

    while (true) {
        batchCount++;

        uint64_t blocks64 = (config.batchSize + threads - 1) / threads;

        int buf = activeBuffer;

        HIP_CHECK(hipMemsetAsync(res.d_outVerified[buf], 0, sizeof(uint64_t), res.streams[buf]));
        HIP_CHECK(hipMemsetAsync(res.d_outTotalSteps[buf], 0, sizeof(uint64_t), res.streams[buf]));
        HIP_CHECK(hipMemsetAsync(res.d_outMaxSteps[buf], 0, sizeof(uint32_t), res.streams[buf]));
        HIP_CHECK(hipMemsetAsync(res.d_outMaxNumber[buf], 0, sizeof(uint64_t), res.streams[buf]));
        HIP_CHECK(hipMemsetAsync(res.d_foundCounter[buf], 0, sizeof(uint32_t), res.streams[buf]));

        hipLaunchKernelGGL(collatzKernel, dim3((unsigned int)blocks64), dim3(threads), 0, res.streams[buf],
                           config.maxSteps, config.batchSize, startValue,
                           res.d_jumpTable,
                           res.d_outVerified[buf], res.d_outTotalSteps[buf],
                           res.d_outMaxSteps[buf], res.d_outMaxNumber[buf],
                           res.d_foundCounter[buf]);
        HIP_CHECK(hipGetLastError());

        if (hasPending) {
            int prevBuf = (buf == 0) ? 1 : 0;
            HIP_CHECK(hipStreamSynchronize(res.streams[prevBuf]));

            uint64_t h_outVerified, h_outTotalSteps, h_outMaxNumber;
            uint32_t h_outMaxSteps;
            uint32_t h_foundCounter;

            HIP_CHECK(hipMemcpy(&h_outVerified, res.d_outVerified[prevBuf], sizeof(uint64_t), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&h_outTotalSteps, res.d_outTotalSteps[prevBuf], sizeof(uint64_t), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&h_outMaxSteps, res.d_outMaxSteps[prevBuf], sizeof(uint32_t), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&h_outMaxNumber, res.d_outMaxNumber[prevBuf], sizeof(uint64_t), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&h_foundCounter, res.d_foundCounter[prevBuf], sizeof(uint32_t), hipMemcpyDeviceToHost));

            if (h_foundCounter != 0) {
                uint32_t realSteps = verifyNumberCPU(h_outMaxNumber, config.maxSteps);
                if (realSteps >= config.maxSteps) {
                    foundCounterExample = true;
                    counterExampleNumber = h_outMaxNumber;
                    printf("\n[!!!] REAL COUNTEREXAMPLE FOUND: %llu\n", (unsigned long long)counterExampleNumber);
                    break;
                }
            }

            totalVerified += h_outVerified;
            totalSteps += h_outTotalSteps;
            totalRealVerified += config.batchSize * 2ull;
            if (h_outMaxSteps > maxSteps) {
                maxSteps = h_outMaxSteps;
                maxStepsNumber = h_outMaxNumber;
            }
            currentNumber += config.batchSize * 2ull;
            startValue = currentNumber;
            hasPending = false;
        }

        activeBuffer = (activeBuffer + 1) % 2;
        hasPending = true;

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - lastDisplay).count();

        if (elapsed >= 0.3) {
            double realSpeed = (double)(totalRealVerified - lastVerified) / elapsed;
            double totalElapsed = std::chrono::duration<double>(now - startTime).count();

            printf("\033[H");
            printf("============================================================\n");
            printf("  AMD HIP Collatz Verifier (corrected)\n");
            printf("  Speed: %.0f numbers/sec\n", realSpeed);
            printf("============================================================\n\n");
            printf("  Batch:      %llu\n", (unsigned long long)batchCount);
            printf("  Verified to: %llu\n", (unsigned long long)currentNumber);
            printf("  Total:      %llu\n", (unsigned long long)totalVerified);
            printf("  Max Steps:  %u (number %llu)\n", maxSteps, (unsigned long long)maxStepsNumber);
            printf("  Time:       %s\n", formatTime(totalElapsed).c_str());
            printf("============================================================\n");
            fflush(stdout);

            lastDisplay = now;
            lastVerified = totalRealVerified;
        }

        if (batchCount % config.saveInterval == 0) {
            Checkpoint newCp = {CHECKPOINT_MAGIC, CHECKPOINT_VERSION,
                               currentNumber, totalVerified, totalSteps,
                               maxSteps, maxStepsNumber, batchCount, 0, startValue, totalRealVerified};
            saveCheckpoint(newCp);
        }

        if (currentNumber > UINT64_MAX - config.batchSize * 2ull) {
            printf("\n[Warning] Reached 64-bit limit\n");
            break;
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double totalSeconds = std::chrono::duration<double>(endTime - startTime).count();

    printf("\n\n============================================================\n");
    printf("  VERIFICATION COMPLETE\n");
    printf("============================================================\n");
    printf("[OK] Verified up to: %llu\n", (unsigned long long)currentNumber);
    printf("[OK] Total verified: %llu\n", (unsigned long long)totalVerified);
    printf("[Best] Max steps: %u (number %llu)\n", maxSteps, (unsigned long long)maxStepsNumber);
    printf("[Time] Total time: %s\n", formatTime(totalSeconds).c_str());

    if (totalSeconds > 0) {
        double avgRealSpeed = totalRealVerified / totalSeconds;
        printf("[Speed] Real speed: %d numbers/sec\n", (int)avgRealSpeed);
    }

    cleanupGPU(res);
    delete[] h_jumpTable;

    return 0;
}
