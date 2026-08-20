# Collatz Conjecture GPU Verifier – AMD HIP Edition

**AMD Radeon GPU | ~35 Billion numbers/sec | Mathematically Exact Verification**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## About

A high-performance, mathematically exact GPU verifier for the Collatz Conjecture, built with AMD HIP.

The Collatz Conjecture (or the `3n+1` problem) states that for any positive integer, repeatedly applying:
- if even, divide by 2
- if odd, multiply by 3 and add 1

will eventually reach 1. This project leverages GPU parallelism to verify billions of numbers per second.

---

## Performance

| Version | Key Optimization | Speed (numbers/sec) |
| :--- | :--- | :--- |
| Baseline | k=20, separate arrays, full-range verify | ~5.9 B |
| v1 | Odd-only verification + eliminated numbers buffer | ~10.3 B |
| v2 | Eliminated results buffer + inline warp reduction | ~13.1 B |
| v3 | AoS merged table (24B entry) + 512M batch | ~18.5 B |
| v4 | Compressed entry (16B, mul in uint32) | ~32.0 B |
| v5 | Bit-packed entry (8B: add\|mul\|steps) + k=18 | ~34.5 B |
| **Final** | Software pipeline + 128-bit multiply + 2x unroll + 1024 threads/block | **~35 B** |

**Total speedup: ~6x from baseline.**

---

## Key Features

### Mathematically Exact Acceleration

Uses the Terras/Everett k-step acceleration: the T map `T(n) = n/2` (even) or `(3n+1)/2` (odd), applied exactly `k` times, has the closed form:

```
T^k(n) = (mul(r) * n + add(r)) >> k,   r = n mod 2^k
```

where `mul(r) = 3^(# odd steps)` and `add(r)` is an additive constant, both depending only on `r`. This is exact for every n — not an approximation.

The multiply `mul(r) * n + add` is computed with `unsigned __int128` to avoid overflow for large n (>= 2^32), then right-shifted by k bits.

### Odd-Only Verification

Even numbers `n = 2m` reduce to `m` in one step. Since `m < n` always falls in the already-verified range, only odd numbers need direct computation. This halves the work.

### 8-Byte Bit-Packed Table Entry

Each table entry packs `add(29 bits) | mul(29 bits) | steps(6 bits)` into a single `uint64_t`, minimizing L2 cache bandwidth. The k=18 table is only 2 MB and fits entirely in L2 cache.

### GPU-Side Reduction

Statistics (verified count, total steps, max steps, counterexample detection) are aggregated on the GPU via warp-level shuffle reduction with one atomic per warp. No intermediate per-number results array is needed.

The max-steps record uses a CAS loop on the associated number to prevent a lower-steps warp from overwriting it after a higher-steps warp has set the record.

### Accurate Step Statistics

Verification status and step count are tracked as separate variables. When a number drops below the already-verified range, it is marked verified with the actual number of acceleration steps taken — not a placeholder value. This keeps total/average step statistics meaningful.

### Double Buffering & Async Streams

Two GPU buffers with HIP streams overlap computation with result transfer, maximizing utilization.

### Checkpoint & Resume

Progress is automatically saved to `checkpoint.bin` every 10 batches. The binary format includes a magic number (`0x43545A4C`, "CTZL") and a version field, so future struct changes will not silently corrupt old checkpoints — an invalid or outdated file is detected and the verifier starts fresh. Stop and resume anytime.

### Counterexample Detection

If a number exceeds the step limit, it is re-verified on the CPU to eliminate false positives.

---

## System Requirements

| Item | Requirement |
| :--- | :--- |
| OS | Windows 10 / 11 |
| GPU | AMD Radeon RX 6000 / 7000 / 9000 series (RDNA 2/3/4) |
| Driver | AMD ROCm 7.1 or newer |
| Compiler | Visual Studio 2022 (with HIP support) |

---

## Project Structure

```
Collatz-GPU-Accelerator/
├── src/
│   ├── collatz.hip    # GPU kernel (HIP/C++)
│   └── collatz.cpp    # Host code (table builder, scheduler)
├── build.bat          # Build script (Windows)
├── run.bat            # Run script with menu
├── collatz_amd.exe    # Compiled binary
├── checkpoint.bin     # Progress checkpoint (auto-generated)
├── LICENSE
└── README.md
```

---

## Quick Start

1. Run `build.bat` to compile.
2. Run `run.bat` to start the verifier.
3. To start fresh, delete `checkpoint.bin`.

---

## Changelog

### v2.1 — Correctness & Maintenance Fixes

- **[Severe] Fixed 64-bit multiplication overflow.** The kernel previously split `mul * n + add` into manual 32-bit halves and mishandled the carry between them, producing wrong results for n >= 2^32. Replaced with `unsigned __int128` (matching the host-side `verifyNumberCPU`).
- **[Medium] Fixed step-count statistics.** When n dropped below `startValue`, `result` was set to `1` as a placeholder and then counted as one step. Verification status and actual step count are now separate variables; `totalSteps` reflects real acceleration steps.
- **[Medium] Fixed max-steps number race condition.** `outMaxNumber` was updated with a bare `atomicExch` after `atomicMax` on `outMaxSteps`, allowing a lower-steps warp to overwrite the number after a higher-steps warp set it. Replaced with a CAS loop that re-checks the current max steps before committing.
- **[Minor] Versioned checkpoint format.** Added `magic` and `version` fields to the `Checkpoint` struct. Old or corrupted checkpoint files are detected and the verifier starts from scratch instead of silently using garbage data.
- **[Minor] Unified naming.** The counterexample flag buffer is now consistently named `foundCounter` across host and kernel code.
- **Tuning.** Increased threads per block from 512 to 1024 to reduce block-scheduling overhead.

---

## Optimization Notes

The final version is **L2-bandwidth bound**. At ~35B numbers/sec with ~6 jumps per number and 8 bytes per jump, it consumes ~1.7 TB/s of L2 bandwidth. Increasing k beyond 18 causes the table to overflow L2 and reduces performance. All micro-optimizations (loop unrolling, software pipelining, register tuning) yield <2% improvement at this point, confirming the bandwidth bottleneck.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Contributing

Issues, bug reports, and pull requests are welcome. If you find a genuine counterexample, that would be the discovery of the century.

---

## Contact

For questions, please open an Issue on GitHub.

---

**Made with math and parallelism by octopodiformes**
