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
| **Final** | Software pipeline + manual 32x64 mul + 2x unroll | **~35 B** |

**Total speedup: ~6x from baseline.**

---

## Key Features

### Mathematically Exact Acceleration

Uses the Terras/Everett k-step acceleration: the T map `T(n) = n/2` (even) or `(3n+1)/2` (odd), applied exactly `k` times, has the closed form:

```
T^k(n) = (mul(r) * n + add(r)) >> k,   r = n mod 2^k
```

where `mul(r) = 3^(# odd steps)` and `add(r)` is an additive constant, both depending only on `r`. This is exact for every n — not an approximation.

### Odd-Only Verification

Even numbers `n = 2m` reduce to `m` in one step. Since `m < n` always falls in the already-verified range, only odd numbers need direct computation. This halves the work.

### 8-Byte Bit-Packed Table Entry

Each table entry packs `add(29 bits) | mul(29 bits) | steps(6 bits)` into a single `uint64_t`, minimizing L2 cache bandwidth. The k=18 table is only 2 MB and fits entirely in L2 cache.

### GPU-Side Reduction

Statistics (verified count, total steps, max steps, counterexample detection) are aggregated on the GPU via warp-level shuffle reduction with one atomic per warp. No intermediate per-number results array is needed.

### Double Buffering & Async Streams

Two GPU buffers with HIP streams overlap computation with result transfer, maximizing utilization.

### Checkpoint & Resume

Progress is automatically saved to `checkpoint.bin` every 10 batches. Stop and resume anytime.

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
