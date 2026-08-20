/**
 * collatz.hip - High-performance Collatz GPU kernel (AMD HIP)
 *
 * Exact k-step Terras/Everett acceleration:
 *   T^k(n) = (mul(r) * n + add(r)) >> k,   r = n mod 2^k
 * where mul(r)=3^(#odd steps), add(r)=additive constant, both depend only on r.
 *
 * Optimizations applied:
 *   - Odd-only verification (evens reduce via n/2)
 *   - 8-bit packed jump table entry: add(29b)|mul(29b)|steps(6b) = 64 bits
 *   - k=18 table (2 MB) fits entirely in L2 cache
 *   - Software-pipelined, 2x-unrolled loop with 128-bit multiply
 *   - Inline warp-level shuffle reduction (one atomic per warp)
 *   - No intermediate results array; stats updated directly via atomics
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

#define TABLE_BITS 18u
#define TABLE_SIZE (1u << TABLE_BITS)
#define TABLE_MASK (TABLE_SIZE - 1u)

#define ADD_MASK   0x1FFFFFFFull
#define MUL_SHIFT  29
#define STEP_SHIFT 58

extern "C" __global__ void collatzKernel(
    uint32_t maxStepsLimit,
    uint64_t numElements,
    uint64_t startValue,
    const uint64_t* __restrict__ jumpTable,
    uint64_t* __restrict__ outVerified,
    uint64_t* __restrict__ outTotalSteps,
    uint32_t* __restrict__ outMaxSteps,
    uint64_t* __restrict__ outMaxNumber,
    uint32_t* __restrict__ foundCounter
) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= numElements) return;

    uint64_t n = startValue + idx * 2ull;
    uint64_t orig = n;

    // Separate semantics: whether the number is verified, and the actual
    // number of acceleration steps taken.  Previously a single "result"
    // variable carried both, causing stats distortion when n dropped below
    // startValue (result was set to 1 as a placeholder, not real steps).
    bool isVerified = false;
    bool isCounterexample = false;
    uint32_t actualSteps = 0;

    if (n <= startValue) {
        // First element (idx == 0) is already in the verified range.
        isVerified = true;
        actualSteps = 0;
    } else {
        uint32_t steps = 0;
        // Software pipeline: prefetch first entry.
        uint32_t r = (uint32_t)(n & TABLE_MASK);
        uint64_t packed = jumpTable[r];

        while (n > startValue && steps < maxStepsLimit) {
            // --- Jump 1 ---
            uint32_t add  = (uint32_t)(packed & ADD_MASK);
            uint32_t mul  = (uint32_t)((packed >> MUL_SHIFT) & ADD_MASK);
            uint32_t jmp  = (uint32_t)(packed >> STEP_SHIFT);
            // Use 128-bit multiply to avoid the carry-overflow bug in the
            // old manual 32x64 split.  The full product mul*n+add can exceed
            // 64 bits, and shifting the truncated product gave wrong results
            // for n >= 2^32.
            n = (uint64_t)(((unsigned __int128)mul * n + add) >> TABLE_BITS);
            steps += jmp;
            if (n <= startValue || steps >= maxStepsLimit) break;

            // Prefetch entry for jump 2
            r = (uint32_t)(n & TABLE_MASK);
            packed = jumpTable[r];

            // --- Jump 2 ---
            add  = (uint32_t)(packed & ADD_MASK);
            mul  = (uint32_t)((packed >> MUL_SHIFT) & ADD_MASK);
            jmp  = (uint32_t)(packed >> STEP_SHIFT);
            n = (uint64_t)(((unsigned __int128)mul * n + add) >> TABLE_BITS);
            steps += jmp;

            // Prefetch entry for next iteration
            if (n > startValue && steps < maxStepsLimit) {
                r = (uint32_t)(n & TABLE_MASK);
                packed = jumpTable[r];
            } else {
                break;
            }
        }

        if (n <= startValue) {
            isVerified = true;
            actualSteps = steps;
        } else if (steps >= maxStepsLimit) {
            isCounterexample = true;
            actualSteps = steps;
        } else {
            // Defensive: should not happen; treat as verified.
            isVerified = true;
            actualSteps = steps;
        }
    }

    // ---- Inline warp-level reduction ----
    uint64_t localVerified = 0;
    uint64_t localSteps = 0;
    uint32_t localMaxSteps = 0;
    uint64_t localMaxNumber = 0;
    uint32_t localFound = 0;
    uint64_t localFoundNumber = 0;

    if (isCounterexample) {
        localFound = 1u;
        localFoundNumber = orig;
    } else if (isVerified) {
        localVerified = 1u;
        localSteps = actualSteps;
        localMaxSteps = actualSteps;
        localMaxNumber = orig;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        localVerified += __shfl_down(localVerified, offset);
        localSteps    += __shfl_down(localSteps, offset);

        uint32_t otherMaxSteps  = __shfl_down(localMaxSteps, offset);
        uint64_t otherMaxNumber = __shfl_down(localMaxNumber, offset);
        if (otherMaxSteps > localMaxSteps) {
            localMaxSteps = otherMaxSteps;
            localMaxNumber = otherMaxNumber;
        }

        uint32_t otherFound       = __shfl_down(localFound, offset);
        uint64_t otherFoundNumber = __shfl_down(localFoundNumber, offset);
        if (otherFound) {
            localFound = 1u;
            localFoundNumber = otherFoundNumber;
        }
    }

    int lane = threadIdx.x & (warpSize - 1);
    if (lane == 0) {
        if (localFound) {
            atomicExch(foundCounter, 1u);
            // Store the suspected counterexample in outMaxNumber.  The host
            // reads this buffer for CPU re-verification when foundCounter != 0.
            atomicExch((unsigned long long*)outMaxNumber, (unsigned long long)localFoundNumber);
        }
        if (localVerified) atomicAdd((unsigned long long*)outVerified, (unsigned long long)localVerified);
        if (localSteps)    atomicAdd((unsigned long long*)outTotalSteps, (unsigned long long)localSteps);

        // Atomically update the global max steps and its associated number.
        // atomicMax on outMaxSteps is safe, but outMaxNumber must be updated
        // with a CAS loop: a warp with fewer steps (whose atomicMax returned
        // a smaller oldMax) could otherwise overwrite the number after a
        // higher-steps warp has set it, leaving outMaxSteps and outMaxNumber
        // inconsistent.
        uint32_t oldMax = atomicMax(outMaxSteps, localMaxSteps);
        if (localMaxSteps > oldMax) {
            uint64_t expected;
            do {
                expected = *outMaxNumber;
                // Abort if another warp has since set a higher step count.
                if (*(volatile uint32_t*)outMaxSteps > localMaxSteps) break;
            } while (atomicCAS((unsigned long long*)outMaxNumber,
                                (unsigned long long)expected,
                                (unsigned long long)localMaxNumber) != (unsigned long long)expected);
        }
    }
}
