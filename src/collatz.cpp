/**
 * collatz.cpp - Corrected Collatz verifier host code
 *
 * Fixes vs. the original:
 *   - Replaced the mathematically invalid "modular residue" skip table
 *     with an exact k-step Collatz acceleration table (see collatz.hip
 *     for the derivation). This is the single most important fix: the
 *     old code was not actually verifying ~95% of numbers.
 *   - Removed the periodic hipDeviceReset() (very expensive, and not
 *     needed - alloc/free was already paired correctly).
 *   - Replaced 5 single-thread "reset" kernel launches per batch with
 *     hipMemsetAsync.
 *   - Reduced default batch size to fit comfortably in 16GB with a
 *     large safety margin (previous default used 12+GB of VRAM by itself).
 *
 * IMPORTANT: any checkpoint.bin produced by the old binary encodes
 * progress that was never actually verified for most numbers. Delete
 * checkpoint.bin before running this version.
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

struct Config {
    uint64_t batchSize = 256ull * 1024 * 1024;   // 256M numbers/batch (~6GB VRAM w/ double buffer)
    uint32_t maxSteps = 100000;
    int saveInterval = 10;
};

struct Checkpoint {
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

Checkpoint loadCheckpoint() {
    Checkpoint cp = {1, 0, 0, 0, 1, 0, 0, 0, 0};
    std::ifstream file("checkpoint.bin", std::ios::binary);
    if (file.is_open()) {
        file.read((char*)&cp, sizeof(Checkpoint));
        file.close();
    }
    return cp;
}

void saveCheckpoint(const Checkpoint& cp) {
    std::ofstream file("checkpoint.bin", std::ios::binary);
    if (file.is_open()) {
        file.write((char*)&cp, sizeof(Checkpoint));
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

uint32_t calcSmallSteps(uint64_t n) {
    uint32_t steps = 0;
    while (n > 1 && steps < 100000) {
        if (n & 1) n = n * 3 + 1;
        else n = n >> 1;
        steps++;
    }
    return steps;
}

// ============================================================
// Build the exact k-step acceleration table.
// T(n) = n/2 (even) or (3n+1)/2 (odd). After TABLE_BITS applications:
//   T^k(n) = (mul(r)*n + add(r)) >> k,   r = n mod 2^k
// mul/add/steps depend only on r - proven by induction, see collatz.hip.
// ============================================================
static const uint32_t TABLE_BITS = 20;
static const uint32_t TABLE_SIZE = 1u << TABLE_BITS;

void buildJumpTable(uint64_t* mul, uint64_t* add, uint32_t* steps) {
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
        mul[r] = (uint64_t)curMul;
        add[r] = (uint64_t)curAdd;
        steps[r] = rawSteps;
    }
}

extern "C" {
    __global__ void collatzKernel(
        uint64_t* numbers, uint32_t* results,
        uint32_t maxStepsLimit, uint64_t numElements,
        uint64_t startValue, const uint32_t* smallSteps,
        const uint64_t* jumpMul, const uint64_t* jumpAdd, const uint32_t* jumpSteps
    );
    __global__ void generateNumbersKernel(uint64_t* numbers, uint64_t startNumber, uint64_t numElements);
    __global__ void reduceResultsKernel(
        uint32_t* results, uint64_t* numbers, uint64_t numElements,
        uint64_t* outVerified, uint64_t* outTotalSteps,
        uint32_t* outMaxSteps, uint64_t* outMaxNumber,
        uint32_t* outFoundCounter
    );
}

struct GPUResources {
    uint64_t* d_numbers[2];
    uint32_t* d_results[2];
    uint64_t* d_outVerified[2];
    uint64_t* d_outTotalSteps[2];
    uint32_t* d_outMaxSteps[2];
    uint64_t* d_outMaxNumber[2];
    uint32_t* d_outFound[2];
    hipStream_t streams[2];

    uint32_t* d_smallSteps;
    uint64_t* d_jumpMul;
    uint64_t* d_jumpAdd;
    uint32_t* d_jumpSteps;

    uint32_t* h_smallSteps;
    uint64_t* h_jumpMul;
    uint64_t* h_jumpAdd;
    uint32_t* h_jumpSteps;

    int SMALL_TABLE_SIZE;
};

void cleanupGPU(GPUResources& res) {
    for (int i = 0; i < 2; i++) {
        if (res.d_numbers[i]) hipFree(res.d_numbers[i]);
        if (res.d_results[i]) hipFree(res.d_results[i]);
        if (res.d_outVerified[i]) hipFree(res.d_outVerified[i]);
        if (res.d_outTotalSteps[i]) hipFree(res.d_outTotalSteps[i]);
        if (res.d_outMaxSteps[i]) hipFree(res.d_outMaxSteps[i]);
        if (res.d_outMaxNumber[i]) hipFree(res.d_outMaxNumber[i]);
        if (res.d_outFound[i]) hipFree(res.d_outFound[i]);
        if (res.streams[i]) hipStreamDestroy(res.streams[i]);
    }
    if (res.d_smallSteps) hipFree(res.d_smallSteps);
    if (res.d_jumpMul) hipFree(res.d_jumpMul);
    if (res.d_jumpAdd) hipFree(res.d_jumpAdd);
    if (res.d_jumpSteps) hipFree(res.d_jumpSteps);
}

void initGPU(GPUResources& res, Config& config) {
    for (int i = 0; i < 2; i++) {
        HIP_CHECK(hipMalloc(&res.d_numbers[i], config.batchSize * sizeof(uint64_t)));
        HIP_CHECK(hipMalloc(&res.d_results[i], config.batchSize * sizeof(uint32_t)));
        HIP_CHECK(hipMalloc(&res.d_outVerified[i], sizeof(uint64_t)));
        HIP_CHECK(hipMalloc(&res.d_outTotalSteps[i], sizeof(uint64_t)));
        HIP_CHECK(hipMalloc(&res.d_outMaxSteps[i], sizeof(uint32_t)));
        HIP_CHECK(hipMalloc(&res.d_outMaxNumber[i], sizeof(uint64_t)));
        HIP_CHECK(hipMalloc(&res.d_outFound[i], sizeof(uint32_t)));
        HIP_CHECK(hipStreamCreate(&res.streams[i]));
    }

    HIP_CHECK(hipMalloc(&res.d_smallSteps, res.SMALL_TABLE_SIZE * sizeof(uint32_t)));
    HIP_CHECK(hipMemcpy(res.d_smallSteps, res.h_smallSteps, res.SMALL_TABLE_SIZE * sizeof(uint32_t), hipMemcpyHostToDevice));

    HIP_CHECK(hipMalloc(&res.d_jumpMul, TABLE_SIZE * sizeof(uint64_t)));
    HIP_CHECK(hipMemcpy(res.d_jumpMul, res.h_jumpMul, TABLE_SIZE * sizeof(uint64_t), hipMemcpyHostToDevice));

    HIP_CHECK(hipMalloc(&res.d_jumpAdd, TABLE_SIZE * sizeof(uint64_t)));
    HIP_CHECK(hipMemcpy(res.d_jumpAdd, res.h_jumpAdd, TABLE_SIZE * sizeof(uint64_t), hipMemcpyHostToDevice));

    HIP_CHECK(hipMalloc(&res.d_jumpSteps, TABLE_SIZE * sizeof(uint32_t)));
    HIP_CHECK(hipMemcpy(res.d_jumpSteps, res.h_jumpSteps, TABLE_SIZE * sizeof(uint32_t), hipMemcpyHostToDevice));
}

int main() {
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipDeviceSetLimit(hipLimitMallocHeapSize, 14ULL * 1024 * 1024 * 1024));

    printf("\033[2J\033[H");
    printf("============================================================\n");
    printf("  AMD HIP Collatz Verifier (corrected)\n");
    printf("  Exact k=%u bit acceleration table\n", TABLE_BITS);
    printf("============================================================\n\n");

    const int SMALL_TABLE_SIZE = 65537;
    uint32_t* h_smallSteps = new uint32_t[SMALL_TABLE_SIZE];
    printf("[Info] Building small number lookup table (0-65536)...\n");
    fflush(stdout);
    for (int i = 0; i < SMALL_TABLE_SIZE; i++) {
        h_smallSteps[i] = calcSmallSteps(i);
    }

    printf("[Info] Building exact k=%u acceleration table (%u entries)...\n", TABLE_BITS, TABLE_SIZE);
    fflush(stdout);
    uint64_t* h_jumpMul = new uint64_t[TABLE_SIZE];
    uint64_t* h_jumpAdd = new uint64_t[TABLE_SIZE];
    uint32_t* h_jumpSteps = new uint32_t[TABLE_SIZE];
    buildJumpTable(h_jumpMul, h_jumpAdd, h_jumpSteps);
    printf("[Info] All tables ready!\n\n");

    Config config;
    GPUResources res;
    res.SMALL_TABLE_SIZE = SMALL_TABLE_SIZE;
    res.h_smallSteps = h_smallSteps;
    res.h_jumpMul = h_jumpMul;
    res.h_jumpAdd = h_jumpAdd;
    res.h_jumpSteps = h_jumpSteps;

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

    printf("[Info] Batch size: %lluM numbers/batch (x2 buffers)\n", (unsigned long long)(config.batchSize / (1024*1024)));
    printf("[Info] Acceleration table: %u bits (~%.1f MB)\n", TABLE_BITS,
           (double)(TABLE_SIZE * (8+8+4)) / (1024*1024));
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

    const int threads = 256;

    printf("Running...\n\n");

    while (true) {
        batchCount++;

        uint64_t batchStart = currentNumber;
        uint64_t blocks64 = (config.batchSize + threads - 1) / threads;

        int buf = activeBuffer;

        hipLaunchKernelGGL(generateNumbersKernel, dim3((unsigned int)blocks64), dim3(threads), 0, res.streams[buf],
                           res.d_numbers[buf], batchStart, config.batchSize);
        HIP_CHECK(hipGetLastError());

        HIP_CHECK(hipMemsetAsync(res.d_outVerified[buf], 0, sizeof(uint64_t), res.streams[buf]));
        HIP_CHECK(hipMemsetAsync(res.d_outTotalSteps[buf], 0, sizeof(uint64_t), res.streams[buf]));
        HIP_CHECK(hipMemsetAsync(res.d_outMaxSteps[buf], 0, sizeof(uint32_t), res.streams[buf]));
        HIP_CHECK(hipMemsetAsync(res.d_outMaxNumber[buf], 0, sizeof(uint64_t), res.streams[buf]));
        HIP_CHECK(hipMemsetAsync(res.d_outFound[buf], 0, sizeof(uint32_t), res.streams[buf]));

        hipLaunchKernelGGL(collatzKernel, dim3((unsigned int)blocks64), dim3(threads), 0, res.streams[buf],
                           res.d_numbers[buf], res.d_results[buf],
                           config.maxSteps, config.batchSize, startValue,
                           res.d_smallSteps, res.d_jumpMul, res.d_jumpAdd, res.d_jumpSteps);
        HIP_CHECK(hipGetLastError());

        hipLaunchKernelGGL(reduceResultsKernel, dim3((unsigned int)blocks64), dim3(threads), 0, res.streams[buf],
                           res.d_results[buf], res.d_numbers[buf], config.batchSize,
                           res.d_outVerified[buf], res.d_outTotalSteps[buf],
                           res.d_outMaxSteps[buf], res.d_outMaxNumber[buf],
                           res.d_outFound[buf]);
        HIP_CHECK(hipGetLastError());

        if (hasPending) {
            int prevBuf = (buf == 0) ? 1 : 0;
            HIP_CHECK(hipStreamSynchronize(res.streams[prevBuf]));

            uint64_t h_outVerified, h_outTotalSteps, h_outMaxNumber;
            uint32_t h_outMaxSteps;
            uint32_t h_outFound;

            HIP_CHECK(hipMemcpy(&h_outVerified, res.d_outVerified[prevBuf], sizeof(uint64_t), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&h_outTotalSteps, res.d_outTotalSteps[prevBuf], sizeof(uint64_t), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&h_outMaxSteps, res.d_outMaxSteps[prevBuf], sizeof(uint32_t), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&h_outMaxNumber, res.d_outMaxNumber[prevBuf], sizeof(uint64_t), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&h_outFound, res.d_outFound[prevBuf], sizeof(uint32_t), hipMemcpyDeviceToHost));

            if (h_outFound != 0) {
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
            totalRealVerified += config.batchSize;
            if (h_outMaxSteps > maxSteps) {
                maxSteps = h_outMaxSteps;
                maxStepsNumber = h_outMaxNumber;
            }
            currentNumber += config.batchSize;
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
            Checkpoint newCp = {currentNumber, totalVerified, totalSteps,
                               maxSteps, maxStepsNumber, batchCount, 0, startValue, totalRealVerified};
            saveCheckpoint(newCp);
        }

        if (currentNumber > UINT64_MAX - config.batchSize) {
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
    delete[] h_smallSteps;
    delete[] h_jumpMul;
    delete[] h_jumpAdd;
    delete[] h_jumpSteps;

    return 0;
}