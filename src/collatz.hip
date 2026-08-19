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
 *   - Software-pipelined, 2x-unrolled loop with manual 32x64 multiply
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
    uint32_t* __restrict__ outFoundCounter
) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= numElements) return;

    uint64_t n = startValue + idx * 2ull;
    uint32_t result;

    if (n <= startValue) {
        result = 1u;
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
            uint64_t n_lo = n & 0xFFFFFFFFull;
            uint64_t n_hi = n >> 32;
            uint64_t p_lo = (uint64_t)mul * n_lo + add;
            uint64_t p_hi = (uint64_t)mul * n_hi;
            n = (p_lo >> TABLE_BITS) | (p_hi << (64 - TABLE_BITS));
            steps += jmp;
            if (n <= startValue || steps >= maxStepsLimit) break;

            // Prefetch entry for jump 2
            r = (uint32_t)(n & TABLE_MASK);
            packed = jumpTable[r];

            // --- Jump 2 ---
            add  = (uint32_t)(packed & ADD_MASK);
            mul  = (uint32_t)((packed >> MUL_SHIFT) & ADD_MASK);
            jmp  = (uint32_t)(packed >> STEP_SHIFT);
            n_lo = n & 0xFFFFFFFFull;
            n_hi = n >> 32;
            p_lo = (uint64_t)mul * n_lo + add;
            p_hi = (uint64_t)mul * n_hi;
            n = (p_lo >> TABLE_BITS) | (p_hi << (64 - TABLE_BITS));
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
            result = 1u;
        } else if (steps >= maxStepsLimit) {
            result = 0xFFFFFFFFu;
        } else {
            result = steps;
        }
    }

    // ---- Inline warp-level reduction ----
    uint64_t orig = startValue + idx * 2ull;
    uint64_t localVerified = 0;
    uint64_t localSteps = 0;
    uint32_t localMaxSteps = 0;
    uint64_t localMaxNumber = 0;
    uint32_t localFound = 0;
    uint64_t localFoundNumber = 0;

    if (result == 0xFFFFFFFFu) {
        localFound = 1u;
        localFoundNumber = orig;
    } else if (result > 0u) {
        localVerified = 1u;
        localSteps = result;
        localMaxSteps = result;
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
            atomicExch(outFoundCounter, 1u);
            atomicExch((unsigned long long*)outMaxNumber, (unsigned long long)localFoundNumber);
        }
        if (localVerified) atomicAdd((unsigned long long*)outVerified, (unsigned long long)localVerified);
        if (localSteps)    atomicAdd((unsigned long long*)outTotalSteps, (unsigned long long)localSteps);

        uint32_t oldMax = atomicMax(outMaxSteps, localMaxSteps);
        if (localMaxSteps > oldMax) {
            atomicExch((unsigned long long*)outMaxNumber, (unsigned long long)localMaxNumber);
        }
    }
}
