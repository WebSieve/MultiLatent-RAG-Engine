<div align="center">

  <h1>🚀 MultiLatent-Rag-Engine</h1>
  <p><b>An ultra-efficient long-context inference engine implementing DeepSeek's Multi-head Latent Attention (MLA) & YaRN RoPE scaling from scratch.</b></p>

  <p>
    <img src="https://img.shields.io/badge/KV_Cache_Compression-Up_to_57x-blue?style=for-the-badge&logo=databricks" alt="Compression">
    <img src="https://img.shields.io/badge/Core_Math-Rust_%2B_PyO3-orange?style=for-the-badge&logo=rust" alt="Rust Core">
    <img src="https://img.shields.io/badge/Framework-Pure_NumPy_%2F_PyTorch-green?style=for-the-badge&logo=pytorch" alt="PyTorch">
  </p>
</div>

---

### 🔥 Why MultiLatent-Rag-Engine?

Traditional Transformers hit a brutal memory bandwidth wall during long-context generation because their KV-cache balloons quadratically. **MultiLatent-Rag-Engine** solves this by combining **low-rank matrix factorization** (compressing states into tiny latent vectors) with **YaRN-scaled Rotary Position Embeddings (RoPE)**—all performance-critical matrix math compiled down to bare-metal **Rust** via PyO3 for zero-copy execution.

* **Massive VRAM Savings:** Shrinks the KV footprint drastically compared to standard Multi-Head Attention (MHA).
* **Infinite Context Scaling:** Dynamically blends NTK-by-parts frequency interpolation to stretch context windows smoothly.
* **Zero CUDA Bloat:** Pure, first-principles systems engineering running seamlessly via CPU-optimized Rust and Python glue.
