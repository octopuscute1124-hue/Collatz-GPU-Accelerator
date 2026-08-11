/**
 * collatz.hip - Corrected & optimized Collatz kernel
 *
 * The previous version's "L1 sieve" looked up skipTableL1[n & 65535] where
 * the table was built by running the Collatz trajectory of the RESIDUE
 * r (0..65535) itself and checking whether IT dropped below r. That does
 * NOT tell you anything valid about a large n with n mod 65536 == r,
 * because 3n+1 depends on every bit of n, not just the low 16. In practice
 * that table was "true" for almost every residue, so ~95% of numbers were
 * marked "verified" without ever being computed.
 *
 * This version replaces it with a mathematically exact acceleration:
 * the "T map" T(n) = n/2 (n even), (3n+1)/2 (n odd), applied exactly
 * TABLE_BITS times, has closed form
 *
 *     T^k(n) = (mul(r) * n + add(r)) >> k,     r = n mod 2^k
 *
 * where mul(r) = 3^(# odd steps among the k), add(r) is an additive
 * constant, and both depend ONLY on r. This is exact for every n, not an
 * approximation - it's the standard Terras/Everett acceleration used by
 * real large-scale Collatz verifiers. One jump replaces ~TABLE_BITS
 * individual branchy steps with one 128-bit multiply-add-shift.
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

#define TABLE_BITS 20u
#define TABLE_SIZE (1u << TABLE_BITS)
#define TABLE_MASK (TABLE_SIZE - 1u)

// ============================================================
// Main Collatz Kernel using the exact k-step jump table
// ============================================================
extern "C" __global__ void collatzKernel(
    uint64_t* __restrict__ numbers,
    uint32_t* __restrict__ results,
    uint32_t maxStepsLimit,
    uint64_t numElements,
    uint64_t startValue,
    const uint32_t* __restrict__ smallSteps,  // exact steps for n in [0, 65536]
    const uint64_t* __restrict__ jumpMul,     // 3^oddCount(r), r in [0, TABLE_SIZE)
    const uint64_t* __restrict__ jumpAdd,     // additive constant(r)
    const uint32_t* __restrict__ jumpSteps    // raw Collatz steps consumed by this jump
) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= numElements) return;

    uint64_t n = numbers[idx];

    // Exact lookup for tiny numbers
    if (n <= 65536ull) { results[idx] = smallSteps[n]; return; }

    // Already covered by a previously, fully verified range
    if (n <= startValue) { results[idx] = 1; return; }

    uint32_t steps = 0;
    while (n > 1ull && n > startValue && steps < maxStepsLimit) {
        uint32_t r = (uint32_t)(n & TABLE_MASK);
        unsigned __int128 v = (unsigned __int128)jumpMul[r] * (unsigned __int128)n
                             + (unsigned __int128)jumpAdd[r];
        n = (uint64_t)(v >> TABLE_BITS);
        steps += jumpSteps[r];
    }

    if (n <= startValue) { results[idx] = 1; return; }
    if (n <= 65536ull) { results[idx] = steps + smallSteps[n]; return; }

    if (steps >= maxStepsLimit) {
        results[idx] = 0xFFFFFFFFu;   // possible counterexample -> re-checked on CPU
    } else {
        results[idx] = steps;
    }
}

// ============================================================
// Generate numbers on GPU (no CPU copy)
// ============================================================
extern "C" __global__ void generateNumbersKernel(
    uint64_t* __restrict__ numbers,
    uint64_t startNumber,
    uint64_t numElements
) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= numElements) return;
    numbers[idx] = startNumber + idx;
}

// ============================================================
// Reduction kernel - GPU-side statistics
// Warp-level shuffle reduction first, then ONE atomic per warp
// instead of one atomic per thread (32-64x less atomic contention).
// ============================================================
extern "C" __global__ void reduceResultsKernel(
    uint32_t* __restrict__ results,
    uint64_t* __restrict__ numbers,
    uint64_t numElements,
    uint64_t* __restrict__ outVerified,
    uint64_t* __restrict__ outTotalSteps,
    uint32_t* __restrict__ outMaxSteps,
    uint64_t* __restrict__ outMaxNumber,
    uint32_t* __restrict__ outFoundCounter
) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;

    uint64_t localVerified = 0;
    uint64_t localSteps = 0;
    uint32_t localMaxSteps = 0;
    uint64_t localMaxNumber = 0;
    uint32_t localFound = 0;
    uint64_t localFoundNumber = 0;

    if (idx < numElements) {
        uint32_t steps = results[idx];
        if (steps == 0xFFFFFFFFu) {
            localFound = 1;
            localFoundNumber = numbers[idx];
        } else if (steps > 0) {
            localVerified = 1;
            localSteps = steps;
            localMaxSteps = steps;
            localMaxNumber = numbers[idx];
        }
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
            localFound = 1;
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

        uint32_t oldMax = *outMaxSteps;
        while (localMaxSteps > oldMax) {
            if (atomicCAS(outMaxSteps, oldMax, localMaxSteps) == oldMax) {
                atomicExch((unsigned long long*)outMaxNumber, (unsigned long long)localMaxNumber);
                break;
            }
            oldMax = *outMaxSteps;
        }
    }
}
