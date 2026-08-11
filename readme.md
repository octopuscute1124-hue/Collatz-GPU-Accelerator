# Collatz Conjecture GPU Verifier – HIP Edition (Corrected)

**AMD Radeon 9060 XT | 5.86+ Billion numbers/sec | Exact Verification**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 About

This is a **high-performance, mathematically corrected** GPU verifier for the Collatz Conjecture, built with AMD HIP.

The Collatz Conjecture (or the `3n+1` problem) states that for any positive integer, repeatedly applying:
- if even, divide by 2
- if odd, multiply by 3 and add 1

will eventually lead to the number 1. Despite its simplicity, it remains unproven for all numbers.

This project leverages the massive parallelism of AMD Radeon GPUs to verify billions of numbers per second.

---

## 🚀 Performance Benchmark

| Hardware | Speed | Improvement vs. CPU |
| :--- | :--- | :--- |
| CPU (Single Core) | ~5 Million/sec | 1x |
| **AMD Radeon 9060 XT (This Verifier)** | **> 5.86 Billion/sec** | **~1,170x** |

**Yes, this is over 1,100x faster than a single CPU core.**

---

## ✨ Key Features & v2.0 Critical Fixes

This version (`v2.0`) is a **major correction** over previous public releases.

- **Fixed: Exact k-Step Acceleration Table**  
  The old version used a mathematically invalid "modular residue" skip table, which meant it **did not actually verify ~95% of numbers**. This has been replaced with a proven, exact `k=20` bit acceleration table. Every result is now genuinely verified.

- **Double Buffering & Async Streams**  
  Uses two GPU buffers and HIP streams to completely overlap computation with data transfer, maximizing utilization.

- **Optimized Resource Management**  
  Replaces expensive `hipDeviceReset()` calls with precise `hipMemsetAsync` operations. Default batch size (256M) is tuned for stability on 16GB GPUs.

- **GPU-Accelerated Reduction**  
  Statistics (max steps, counterexample detection) are aggregated directly on the GPU. The CPU only reads a tiny result packet per batch.

- **Checkpoint & Resume**  
  Automatically saves progress to `checkpoint.bin`. You can stop and resume anytime without losing work.

- **Counterexample Detection**  
  If a number exceeds the step limit, it's re-verified on the CPU to eliminate false positives.

---

## 🛠️ System Requirements

| Item | Requirement |
| :--- | :--- |
| OS | Windows 10 / 11 |
| GPU | AMD Radeon RX 6000 / 7000 / 9000 series (RDNA 2/3/4) |
| Driver | AMD ROCm 7.1 or newer |
| Compiler | Visual Studio 2022 (with HIP support) |
| Dependencies | HIP, ROCm |

---

## 📁 Project Structure

Collatz-GPU-Accelerator/

├── src/

│ ├── collatz.hip # GPU Kernel (HIP/C++) – THE CORE LOGIC

│ └── collatz.cpp # Host code (CPU control, table builder)

├── build.bat # Build script (Windows)

├── run.bat # Run script with menu

├── collatz_amd.exe # Compiled binary

├── checkpoint.bin # Progress checkpoint (auto-generated)

├── LICENSE # MIT License

└── README.md # This file

---

## ⚡ Quick Start

1. **Clone or download** this repository.
2. **Build** the executable by running `build.bat`.
3. **Run** the verifier by executing `run.bat`.
4. To **start a fresh verification**, simply delete the `checkpoint.bin` file.

---

📊 Sample Output (v2.0)

============================================================
AMD HIP Collatz Verifier (corrected)
Exact k=20 bit acceleration table
============================================================

[Info] Batch size: 256M numbers/batch (x2 buffers)
[Info] Acceleration table: 20 bits (~20.0 MB)

[Info] Starting from: 78449773248507

Running...
============================================================
AMD HIP Collatz Verifier (corrected)
Speed: 5864780763 numbers/sec
============================================================

Batch: 246025
Verified to: 78496481017851
Total: 78496481017849
Max Steps: 1307 (number 202485402111)
Time: 7.9s
============================================================

---

## ⚠️ Important Note on Checkpoints

**Checkpoint files (`.bin`) created by the old, incorrect version are INVALID for this verifier.**  
If you are upgrading, please delete your old `checkpoint.bin` before running `v2.0`. The program will warn you about this.

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Issues, bug reports, and pull requests are welcome! If you find a genuine counterexample, that would be the discovery of the century.

---

## 📮 Contact

For questions, please open an Issue on GitHub.

---

**Made with 🔢 and a commitment to correctness by octopodiformes**
