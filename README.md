
<!-- <h1 align="center">The GPU Kernel Engineer’s Handbook</h1> -->

<h1 align="center">GPU Kernel Engineering</h1>

<p align="center">
<em>Learn by building and optimizing attention on Blackwell data center GPUs.</em>
</p>

<p align="center">
90-page visual book · 14 kernel progression · 60 diagrams · 94.5% of FlashAttention-4 · Video-generation capstone
</p>


<h2 align="center">
  <a href="https://iaroslavelistratov.github.io/b200-attention/">🎨 Read visual guide</a> ·
  <a href="kernels/">⚙️ Browse kernels</a> ·
  <a href="benchmarks/">📊 Benchmark</a> ·
  <a href="capstone-project/">🚀 Run capstone</a>
</h2>



[![B200 Attention from Scratch](assets/b200-attention-social-card.png)](https://iaroslavelistratov.github.io/b200-attention/)


[Download e-book](https://github.com/IaroslavElistratov/gpu-kernel-engineering/releases/download/book-v1.0/gpu-kernel-engineering.pdf).
PDF edition of the blog.
<!-- The PDF and online visual guide contain the same material. -->


## **Build and understand one of the hardest GPU kernels on the latest hardware.**

Attention sits at the heart of transformer models.
Making it fast is one of the most demanding problems in GPU programming.
Dense attention also forms the basis for sparse-attention kernels used in some modern frontier models.

This book teaches CUDA/PTX kernel engineering through a gradual, step by step explanation of FlashAttention-4.
I think that kernel is a great real-world showcase of most of Blackwell’s machinery.
Along the way, you’ll learn tensor-core programming, memory layouts, TMA, tcgen05, TMEM, warp specialization, pipelining,
and many other optimizations listed below.

Each technique appears when the kernel needs it, so you see not only how it works, but what problem it solves.
The book follows a single kernel, but the techniques you learn extend well beyond attention.

<!--This book uses FlashAttention-4 as a reference to build a high-performance attention kernel step by step, in CUDA/PTX. -->

<!-- todo-low: rm? -->
<!-- A visual, diffable path from scratch to a high-performance NVIDIA B200 kernel. -->


<!-- positioning: PMPP is the textbook; this is the book for after it -->

## **How to follow along**

If you’re comfortable with basic CUDA and matrix multiplication, you have
the background to follow this book. No prior Blackwell experience is required:
I introduce its machinery gradually, with step-by-step explanations and diagrams.

**I recommend reading the [visual guide](https://iaroslavelistratov.github.io/b200-attention/)
or its [PDF edition](https://github.com/IaroslavElistratov/gpu-kernel-engineering/releases/download/book-v1.0/gpu-kernel-engineering.pdf)
with the corresponding kernels open alongside it.**
The guide explains the ideas and design choices; the CUDA/PTX source shows
how they’re implemented.

If you’re new to CUDA, I’ve listed a few introductory resources at the end of this README.


## **Table of contents**

<!--
aids the book like framing; and communicates the deep substance of the work
make the table of contents read as techniques
-->

- **[Part I — Basics](https://iaroslavelistratov.github.io/b200-attention/#part-i--basics)**
  - [Chapter 1 – Baseline Blackwell B200 Attention](https://iaroslavelistratov.github.io/b200-attention/#chapter-1--baseline-blackwell-b200-attention)
    - [A) Q@K.T (explaining K-major)](https://iaroslavelistratov.github.io/b200-attention/#a-qkt-explaining-k-major)
    - [B) Online Softmax (explaining S->P)](https://iaroslavelistratov.github.io/b200-attention/#b-online-softmax-explaining-s-p)
    - [C) P@V (explaining MN-major)](https://iaroslavelistratov.github.io/b200-attention/#c-pv-explaining-mn-major)
  - [Chapter 2 – Move P from SMEM to TMEM](https://iaroslavelistratov.github.io/b200-attention/#chapter-2--move-p-from-smem-to-tmem)
  - [Chapter 3 – Swizzling](https://iaroslavelistratov.github.io/b200-attention/#chapter-3--swizzling)
  - [Chapter 4 – Warp Specialization](https://iaroslavelistratov.github.io/b200-attention/#chapter-4--warp-specialization)
- **[Part II — Coarse Scheduling](https://iaroslavelistratov.github.io/b200-attention/#part-ii--coarse-scheduling)**
  - [Chapter 5 – Two Q Tiles](https://iaroslavelistratov.github.io/b200-attention/#chapter-5--two-q-tiles)
  - [Chapter 6 – Load Pipeline](https://iaroslavelistratov.github.io/b200-attention/#chapter-6--load-pipeline)
  - [Chapter 7 – Compute Pipeline](https://iaroslavelistratov.github.io/b200-attention/#chapter-7--compute-pipeline)
- **[Part III — Hot-Loop Optimizations](https://iaroslavelistratov.github.io/b200-attention/#part-iii--hot-loop-optimizations)**
  - [Chapter 8 – Hardware Approximations](https://iaroslavelistratov.github.io/b200-attention/#chapter-8--hardware-approximations)
  - [Chapter 9 – Software Approximations](https://iaroslavelistratov.github.io/b200-attention/#chapter-9--software-approximations)
  - [Chapter 10 – Register Caching](https://iaroslavelistratov.github.io/b200-attention/#chapter-10--register-caching)
  - [Chapter 11 – Skip O Rescale](https://iaroslavelistratov.github.io/b200-attention/#chapter-11--skip-o-rescale)
  - [Chapter 12 – Split Correction from Softmax](https://iaroslavelistratov.github.io/b200-attention/#chapter-12--split-correction-from-softmax)
  - [Chapter 13 – Early PV](https://iaroslavelistratov.github.io/b200-attention/#chapter-13--early-pv)
- **[Part IV — Persistent Kernels](https://iaroslavelistratov.github.io/b200-attention/#part-iv--persistent-kernels)**
  - [Chapter 14 – Persistent](https://iaroslavelistratov.github.io/b200-attention/#chapter-14--persistent)
- [Minor Optimizations](https://iaroslavelistratov.github.io/b200-attention/#minor-optimizations)
- [**Capstone Project** – Generate Videos!](https://iaroslavelistratov.github.io/b200-attention/#capstone-project--generate-videos)



I developed and refined this CUDA/PTX kernel lineage over several months.
I understand every line of these kernels.
I wrote the book myself, based on that work.

These kernels are intended for teaching and research. They are not a
line by line port of the official FlashAttention-4 implementation.


## Performance progression

![Performance across the B200 attention kernel lineage](assets/results_condensed_context_pct_stock_fa4.png)


## Kernel lineage

[`kernels/`](kernels/) - 14 kernel checkpoints.
Each is organized around one main optimization, so you can
study the explanation and diff the neighboring source files.

## Capstone: generate videos with the kernel you now understand

<!-- <video controls muted src="assets/videos/b200-attention-capstone-demo.mp4"></video> -->
https://github.com/user-attachments/assets/7c30297f-a555-4e4d-923d-d3a8b6defabc



I strongly recommend doing the capstone after reading the article.
It plugs the B200 attention kernel into a video generation model,
where the model calls **YOUR KERNEL (or the kernel you understand)** 1,056 times
for each generated video.

**[Run the video-generation capstone](capstone-project/)**


# Stay in touch for more ML systems

A video walkthrough of the code is coming.

More ML systems articles, videos, and code are coming. Follow on:

[X](https://x.com/iaro_e) ·
[LinkedIn](https://www.linkedin.com/in/iaroslav-elistratov/) ·
[YouTube](https://www.youtube.com/@IaroslavElistratov) ·
[GitHub](https://github.com/IaroslavElistratov) ·
[Google Scholar](https://scholar.google.com/citations?user=PYpBRIkAAAAJ)


---

Introductory CUDA resources.

Either (1) watch the first 5 videos [in this playlist](https://www.youtube.com/playlist?list=PLRRuQYjFhpmvu5ODQoY2l7D0ADgWEcYAX). Don’t worry that these videos are old, they teach GPU-programming fundamentals that still apply today. Or (2) read the first few chapters of the Programming Massively Parallel Processors (PMPP) book.
