# Optimize CUDA Matmul Kernel.

This chapter is understanding of [SIBOEHM](https://siboehm.com/articles/22/CUDA-MMM). Thank you Siboehm for the amazing blog.

- As siboehm mentioned, In this post, I’ll iteratively optimize an implementation of matrix multiplication written in CUDA. My goal is not to build a cuBLAS replacement, but to deeply understand the most important performance characteristics of the GPUs that are used for modern deep learning. This includes coalescing global memory accesses, shared memory caching and occupancy optimizations, among others.You can download the code for all kernels from Github. Also checkout wangzyon’s repo from which I copied the benchmarking setup.

Matrix multiplication on GPUs may currently be the most important algorithm that exists, considering it makes up almost all the FLOPs during the training and inference of large deep-learning models.

| Implementation | KernelGFLOPs/s | Performance relative to cuBLAS |
|----------------|----------------|--------------------------------|
| 1: Naive | 309.0 | 1.3% |
| 2: GMEM Coalescing | 1986.5 | 8.5% |
| 3: SMEM Caching | 2980.3 | 12.8% |
| 4: 1D Blocktiling | 8474.7 | 36.5% |
| 5: 2D Blocktiling | 15971.7 | 68.7% |
| 6: Vectorized Mem Access | 18237.3 | 78.4% |
| 9: Autotuning | 19721.0 | 84.8% |
| 10: Warptiling | 21779.3 | 93.7% |
| 0: cuBLAS | 23249.6 | 100.0% |

Kernel 1: Naive Implementation

In CUDA, every computation is set in a 3 level hierarchy, Each invocation of the CUDA kernel creates a new grid, which consists of multiple blocks, each block consists of upto 1024 individual thread. Threads that are in the same block have access to the same shared memory region (SMEM).

# CUDA Programming Model: A Simple Guide

## 🏙️ Hierarchy Overview

CUDA organizes computation in a three-level hierarchy:

1. **Kernel** (Entire Program)
2. **Grid** (Collection of Blocks)
3. **Block** (Group of Threads)
4. **Thread** (Individual Unit of Computation)

## 🏗️ Structure Visualization


```python
Kernel
│
├── Grid 1
│   ├── Block 1
│   │   ├── Thread 1
│   │   ├── Thread 2
│   │   └── ...
│   │
│   └── Block 2
│       ├── Thread 1
│       ├── Thread 2
│       └── ...
│
└── Grid 2
├── Block 1
│   ├── Thread 1
│   ├── Thread 2
│   └── ...
└── ...
```

## 🔑 Key Concepts
- **Kernel**: The entire CUDA program
- **Grid**: Division of work into blocks
- **Block**: Group of threads that can cooperate
- **Thread**: Individual unit of parallel execution

## 💡 Important Points
- Threads in the same block can communicate and share resources
- Blocks in a grid run independently
- Multiple grids run sequentially

## 🗄️ Shared Memory (SMEM)
- Fast memory shared by threads within a block
- Accessible only to threads in the same block
- Faster than global memory

## ⚠️ Limitations
- Maximum 1024 threads per block
- Number of blocks and grids varies by GPU

## Detailed Explanations (Nature example)

#### The Big Picture: CUDA Hierarchy
Imagine you're organizing a massive cleaning operation for a city. The CUDA hierarchy is like this:
The entire city = Your whole program (Kernel)
Neighborhoods = Grids
Apartment buildings = Blocks
Individual cleaners = Threads

#### Levels of the Hierarchy (from top to bottom):
- Kernel: This is your entire CUDA program. When you run it, it's like starting the cleaning operation for the whole city.
- Grid: Think of this as dividing the city into neighborhoods. Each neighborhood (grid) has its own set of tasks.
- Block: In each neighborhood, you have apartment buildings. Each building (block) has a team of cleaners working together.
- Thread: These are individual cleaners, each doing a specific task.

#### Key Points:
- Threads in the same block can easily communicate and share resources (like cleaners in the same building sharing cleaning supplies).
- Blocks in a grid run independently (like different apartment buildings working on their own schedules).
- You can have multiple grids, but they run one at a time (like cleaning different neighborhoods on different days).

#### Shared Memory (SMEM):
- This is like a supply closet in each apartment building.
- Only cleaners (threads) in that building (block) can access it.
- It's faster to use than going to the central warehouse (global memory).

#### Limitations:
- Each block can have up to 1024 threads (like a maximum of 1024 cleaners per building).
- The number of blocks and grids can vary based on your GPU.

# Visualzing the diagram
![](https://siboehm.com/assets/img/CUDA-MMM/CUDA_thread_hierarchy.png)

Example to Visualize:

#### GRID:
- The leftmost cube represents a Grid.
- Think of the Grid as the entire office building.
- It's divided into smaller units called Blocks.
- gridDim.x, gridDim.y, gridDim.z: These are like the building's dimensions (length, width, height).

#### BLOCK:
- The middle cube represents a Block.
- If the Grid is the building, a Block is like a single floor or department.
- Each Block contains multiple Threads.
- blockDim.x, blockDim.y, blockDim.z: These are the dimensions of the Block (how many Threads in each direction).

#### THREAD:
- The rightmost part represents a single Thread.
- If the Block is a floor, a Thread is like an individual worker.
- Each Thread does a specific part of the computation.
- "A single thread of computation, minding its own business": This means each Thread works independently on its assigned task.

#### Coordinates and Indexing:
- blockIdx.x, blockIdx.y, blockIdx.z: These tell you which Block you're in within the Grid (like floor number in a building).
- threadIdx.x, threadIdx.y, threadIdx.z: These tell you which Thread you're dealing with within a Block (like desk number on a floor).

Imagine you're organizing a massive book sorting task:
- The entire library is your Grid.
- Each floor of the library is a Block.
- Each worker sorting books is a Thread.
- **gridDim** tells you how many floors the library has.
- **blockDim** tells you how many workers are on each floor.
- **blockIdx** tells you which floor you're on.
- **threadIdx** tells you which worker you are on your floor.

Why This Matters:

This structure allows CUDA to organize massive parallel computations.
It can assign different parts of a problem to different Threads, Blocks, and Grids.


## Steps to run the code.
```cuda
nvcc -o sgemm_naive optimize_matmul.cu
./sgemm_naive
```
