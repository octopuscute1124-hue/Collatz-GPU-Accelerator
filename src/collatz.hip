/**
 * collatz.hip - GPU Kernel for Collatz Conjecture Verification
 * 
 * Architecture: AMD Radeon 9060XT (2048 cores, RDNA 3)
 * Performance: 3.74 Billion numbers/sec
 * 
 * This file contains all GPU kernels:
 *   - generate_numbers_kernel : Fill array with sequential numbers
 *   - collatz_kernel          : Compute Collatz sequence steps
 *   - reduce_kernel           : Aggregate statistics on GPU
 *   - reset_kernel            : Zero out a value
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

// ============================================================
// KERNEL: Generate Sequential Numbers
// ============================================================
// Fills an array with numbers: start, start+1, start+2, ...
// ============================================================

extern "C" __global__ void generate_numbers_kernel(
    uint64_t* __restrict__ numbers,  // Output array
    uint64_t start,                  // Starting number
    uint64_t count                   // Number of elements to generate
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    numbers[idx] = start + idx;
}

// ============================================================
// KERNEL: Collatz Computation
// ============================================================
// Computes the number of steps for each number to reach 1.
// Uses early termination to skip numbers that fall below start_value.
// ============================================================

extern "C" __global__ void collatz_kernel(
    const uint64_t* __restrict__ numbers,  // Input: numbers to check
    uint32_t* __restrict__ results,        // Output: step counts
    uint32_t* __restrict__ max_steps,      // Output: global max steps
    uint32_t max_limit,                    // Max steps before timeout
    uint64_t count,                        // Number of elements
    uint64_t start_value                   // Early termination threshold
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t n = numbers[idx];
    uint32_t steps = 0;

    // Stage 1: Fast path - early termination
    // If number drops below start_value, it's already verified
    #pragma unroll 8
    while (n > start_value && steps < max_limit) {
        if (n & 1) {
            n = n * 3 + 1;
        } else {
            n = n >> 1;
        }
        steps++;
    }

    if (n <= start_value) {
        results[idx] = 1;  // Mark as verified (already known)
        return;
    }

    // Stage 2: Full computation
    #pragma unroll 8
    while (n > 1 && steps < max_limit) {
        if (n & 1) {
            n = n * 3 + 1;
        } else {
            n = n >> 1;
        }
        steps++;
    }

    // Stage 3: Check for counterexample
    if (steps >= max_limit && n > 1) {
        results[idx] = 0xFFFFFFFF;  // Counterexample flag
    } else {
        results[idx] = steps;
    }
}

// ============================================================
// KERNEL: Reset Value (32-bit)
// ============================================================

extern "C" __global__ void reset_kernel_32(uint32_t* value) {
    *value = 0;
}

// ============================================================
// KERNEL: Reset Value (64-bit)
// ============================================================

extern "C" __global__ void reset_kernel_64(uint64_t* value) {
    *value = 0;
}

// ============================================================
// KERNEL: GPU Reduction
// ============================================================
// Aggregates statistics on GPU:
//   - Verified count
//   - Total steps
//   - Maximum steps and its number
//   - Counterexample flag
// 
// CPU only reads 5 numbers per batch! (Not 512M!)
// ============================================================

extern "C" __global__ void reduce_kernel(
    const uint32_t* __restrict__ results,   // Input: step counts
    const uint64_t* __restrict__ numbers,   // Input: original numbers
    uint64_t count,                         // Number of elements
    
    uint64_t* __restrict__ out_verified,    // Output: verified count
    uint64_t* __restrict__ out_total_steps, // Output: total steps
    uint32_t* __restrict__ out_max_steps,   // Output: maximum steps
    uint64_t* __restrict__ out_max_number,  // Output: number with max steps
    uint32_t* __restrict__ out_found        // Output: counterexample flag
) {
    // Shared memory for block-level aggregation
    __shared__ uint64_t s_verified;
    __shared__ uint64_t s_total_steps;
    __shared__ uint32_t s_max_steps;
    __shared__ uint64_t s_max_number;
    __shared__ uint32_t s_found;
    
    // Initialize shared memory (only thread 0)
    if (threadIdx.x == 0) {
        s_verified     = 0;
        s_total_steps  = 0;
        s_max_steps    = 0;
        s_max_number   = 0;
        s_found        = 0;
    }
    __syncthreads();
    
    // Each thread processes one element
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    uint32_t steps = results[idx];
    
    // Counterexample detected!
    if (steps == 0xFFFFFFFF) {
        atomicExch(&s_found, 1);
        atomicExch((unsigned long long*)&s_max_number, numbers[idx]);
        return;
    }
    
    // Normal result: aggregate statistics
    if (steps > 0) {
        atomicAdd(&s_verified, 1);
        atomicAdd(&s_total_steps, steps);
        
        // Update max steps (with CAS for correctness)
        uint32_t old_max = s_max_steps;
        while (steps > old_max) {
            if (atomicCAS(&s_max_steps, old_max, steps) == old_max) {
                atomicExch((unsigned long long*)&s_max_number, numbers[idx]);
                break;
            }
            old_max = s_max_steps;
        }
    }
    
    __syncthreads();
    
    // Thread 0 writes block results to global memory
    if (threadIdx.x == 0) {
        // Counterexample
        if (s_found != 0) {
            atomicExch(out_found, 1);
            atomicExch((unsigned long long*)out_max_number, s_max_number);
        }
        
        // Statistics
        atomicAdd((unsigned long long*)out_verified, s_verified);
        atomicAdd((unsigned long long*)out_total_steps, s_total_steps);
        
        // Max steps (with CAS)
        uint32_t old_max = *out_max_steps;
        while (s_max_steps > old_max) {
            if (atomicCAS((unsigned int*)out_max_steps, old_max, s_max_steps) == old_max) {
                atomicExch((unsigned long long*)out_max_number, s_max_number);
                break;
            }
            old_max = *out_max_steps;
        }
    }
}