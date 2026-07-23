```markdown
# 🔢 Collatz Conjecture GPU Accelerator

<div align="center">

**AMD Radeon 9060XT | 3.74 Billion numbers/sec | 2048 Cores**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Windows-blue)](https://github.com/)
[![GPU](https://img.shields.io/badge/GPU-AMD%20ROCm-red)](https://rocm.docs.amd.com/)

</div>

---

## 📖 About

A high-performance GPU-accelerated validator for the **Collatz Conjecture** using AMD HIP.

The Collatz Conjecture is one of the most famous unsolved problems in mathematics:

> For any positive integer n, if n is even, divide by 2; if n is odd, multiply by 3 and add 1. Eventually, it will always reach 1.

This project uses massive GPU parallelism to verify billions of numbers against this rule in a matter of minutes.

---

## 🚀 Performance Benchmark

| Hardware | Speed | Improvement |
|----------|-------|-------------|
| CPU (Single Core) | ~5 Million/sec | 1x |
| **AMD Radeon 9060XT (This Project)** | **3.74 Billion/sec** | **~750x** |

**Yes, this is 750x faster than a single CPU core.**

---

## 📁 Project Structure

```
Collatz-GPU-Accelerator/
├── src/
│   ├── collatz.hip       # GPU Kernel (HIP/C++)
│   └── collatz.cpp       # Host code (CPU control)
├── build.bat             # Build script (Windows)
├── run.bat               # Run script (menu + checkpoint)
├── start.bat             # Quick start
├── collatz_amd.exe       # Compiled binary (after build)
├── checkpoint.bin        # Progress checkpoint (runtime)
├── LICENSE               # MIT License
└── README.md             # This file
```

---

## ⚡ Quick Start

```bash
git clone https://github.com/octopuscute1124-hue/Collatz-GPU-Accelerator.git
cd Collatz-GPU-Accelerator
build.bat
run.bat
```

---

## 🛠️ System Requirements

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 |
| GPU | AMD Radeon RX 6000/7000 series (RDNA 2/3) |
| Driver | AMD ROCm 7.1 or newer |
| Compiler | Visual Studio 2022 |
| Dependencies | HIP, ROCm |

---

## 🎯 Technical Highlights

- **Double Buffering** – Hide transfer latency, maximize GPU utilization
- **Early Termination** – Skip already-verified numbers to save compute
- **GPU Reduction** – Aggregate statistics on GPU; CPU reads only 5 numbers per batch
- **Checkpointing** – Resume from last progress if interrupted
- **Counterexample Detection** – Real-time detection and validation

---

## 📊 Sample Output

```
============================================================
  COLLATZ CONJECTURE GPU VALIDATOR
  Radeon 9060XT | 2048 Cores | 3.74 Billion/sec
============================================================

[GPU] Found 1 AMD GPU(s)
  0: AMD Radeon 9060XT
      Compute Units: 32
      Max Threads/Block: 1024

[Config] Free VRAM: 15 GB
[Config] Batch Size: 512M numbers
[Config] Buffers: 2
[Config] Threads/Block: 256

Running...

============================================================
  COLLATZ GPU VALIDATOR
  Speed: 3740000000 numbers/sec
============================================================

  Batch:      42
  Verified to: 21,504,000,000
  Total:      21,504,000,000
  Max Steps:  523 (number 9,780,657,631)
  Time:       5m 45s
============================================================
```

---

## ⚠️ Disclaimer

This program is for academic research and performance demonstration purposes.  
While the Collatz Conjecture has been verified for an enormous range, it remains mathematically unproven.  
This program does not guarantee finding a counterexample, nor is it guaranteed to be bug-free.

---

## 📄 License

MIT License – use it, modify it, copy it, just keep the copyright notice.

---

## 🤝 Contributing

Issues and Pull Requests are welcome!

---

## 📮 Contact

Open an Issue on GitHub for any questions.

---

Made with 🔢 by octopodiformes
```