# Introduction to OpenMP

## What is OpenMP?

OpenMP (Open Multi-Processing) is an API that supports multi-platform shared memory multiprocessing programming in C, C++, and Fortran. It provides a simple and flexible interface for developing parallel applications on platforms ranging from desktop computers to supercomputers.

## Key Features

- **Simple parallelism**: Use compiler directives (pragmas) to parallelize code.
- **Shared memory model**: Threads share the same address space.
- **Portability**: Supported by many compilers on different platforms.
- **Incremental parallelism**: Start with a serial program and parallelize step by step.

## How OpenMP Works

OpenMP uses compiler directives, runtime library routines, and environment variables to control parallelism. The most common way to use OpenMP is by adding `#pragma omp` directives in your code.

Example:
```cpp
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    // Loop body executed in parallel
}
```

## When to Use OpenMP

- When you have CPU-bound tasks that can be divided into independent subtasks.
- For loops or sections of code that can run concurrently.
- On shared-memory systems (multi-core CPUs).

## Basic Structure

1. **Include the header**:  
   ```cpp
   #include <omp.h>
   ```
2. **Enable OpenMP in your compiler**:  
   - For GCC/Clang: `-fopenmp`
   - For MSVC: `/openmp`
3. **Add OpenMP directives**:  
   Use `#pragma omp ...` before loops or code blocks to parallelize.

---

# OpenMP Directives

## The `parallel` Directive

### Purpose

The `parallel` directive is used to specify a region of code that should be executed by multiple threads in parallel. When a thread encounters a `#pragma omp parallel` directive, it creates a team of threads, and each thread executes the code block independently.

### When to Use

- When you want to execute a block of code concurrently across multiple threads.
- For tasks that can be performed independently and do not require synchronization within the block.
- To quickly parallelize a section of code without splitting it into smaller tasks.

### Syntax Example

```cpp
#pragma omp parallel
{
    // This block is executed by all threads in the team
    int thread_id = omp_get_thread_num();
    printf("Hello from thread %d\n", thread_id);
}
```

### Common Clauses

- **num_threads(n)**: Specifies the number of threads to use.
  ```cpp
  #pragma omp parallel num_threads(4)
  ```
- **private(var)**: Each thread has its own instance of the variable.
- **shared(var)**: The variable is shared among all threads.
- **default(shared|none)**: Sets the default data-sharing attribute.
- **if(expr)**: The region is executed in parallel only if the expression is true.

#### Example with Clauses

```cpp
int x = 10;
#pragma omp parallel num_threads(4) private(x) shared(arr)
{
    x = omp_get_thread_num();
    arr[x] = x * x;
}
```

**Explanation:**
- `num_threads(4)`: Creates 4 threads to execute the parallel region.
- `private(x)`: Each thread has its own private copy of `x`. Changes to `x` inside the parallel region do not affect the original `x` outside or other threads' `x`.
- `shared(arr)`: The array `arr` is shared among all threads, so each thread can write to its own index.
- Inside the block, each thread sets its private `x` to its thread ID (`omp_get_thread_num()`) and writes `x * x` to the corresponding index in the shared array.

This ensures that each thread works independently on its own data, but results are collected in the shared array.

### Summary Table

| Clause         | Purpose                                      |
|----------------|----------------------------------------------|
| num_threads(n) | Set number of threads                        |
| private(var)   | Each thread gets its own copy of `var`       |
| shared(var)    | All threads share the same `var`             |
| default(...)   | Set default data-sharing behavior            |
| if(expr)       | Conditional parallel execution               |

---

## The `critical` Directive

### Purpose

The `critical` directive is used to specify a section of code that must be executed by only one thread at a time. It is typically used to protect updates to shared variables or resources, preventing data races and ensuring correct results when multiple threads might access or modify the same data.

### When to Use

- When multiple threads need to update or access a shared variable or resource.
- When you want to prevent data races in a parallel region.
- When atomic operations are not sufficient (for example, when updating multiple variables together).
- `critical` have same use case like lock with mutex mechanism in C++. It make sure shared resouce have only 1 thread can access at a time.
### Syntax Example

```cpp
#pragma omp parallel num_threads(4) shared(arr)
{
    int idx = 0; // All threads write to the same index for demonstration
    int value = omp_get_thread_num() + 1;
    #pragma omp critical
    {
        arr[idx] += value; // Only one thread updates arr[idx] at a time
        printf("Thread %d updated arr[%d] to %d\n", omp_get_thread_num(), idx, arr[idx]);
    }
}
```

### Common Clauses

- The `critical` directive can be named to allow multiple independent critical sections:
  ```cpp
  #pragma omp critical(name)
  ```

### Explanation

- The code inside the `critical` section is executed by only one thread at a time.
- This prevents simultaneous updates to shared variables, avoiding data races.
- Named critical sections (`#pragma omp critical(name)`) allow different critical regions to be independent, so threads can enter different named critical sections simultaneously.

### Summary Table

| Clause/Usage                | Purpose                                            |
|-----------------------------|---------------------------------------------------|
| `#pragma omp critical`      | Only one thread executes the block at a time      |
| `#pragma omp critical(name)`| Named critical section for independent protection |

---

## The `for` Directive

### Purpose

The `for` directive is used to parallelize loops, distributing loop iterations among multiple threads. It is most effective when each iteration of the loop is independent of the others.

### When to Use

- When you have a loop with independent iterations.
- For data-parallel tasks, such as processing arrays or matrices.
- When you want to utilize multiple CPU cores to speed up computations.

### Syntax Example

```cpp
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    arr[i] = i * i;
}
```

---

### Common Clauses

#### 1. **collapse(n)**
- **Purpose:** Collapses the next `n` nested loops into a single loop with a larger iteration space, allowing OpenMP to parallelize all iterations across threads.
- **Usage:**
  ```cpp
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
          // loop body
      }
  }
  ```
- **Effect:** Treats the two loops as one loop with `N*M` iterations, improving load balancing and parallel efficiency.

#### 2. **schedule(type[, chunk_size])**
- **Purpose:** Controls how loop iterations are divided among threads.
- **Types:**
  - `static`: Divides iterations into equal-sized chunks assigned to threads in order.
  - `dynamic`: Threads grab chunks as they finish previous ones, good for uneven workloads.
  - `guided`: Chunk size starts large and decreases over time.
  - `auto`/`runtime`: Lets the compiler or environment decide.
- **Usage:**
  ```cpp
  #pragma omp parallel for schedule(dynamic, 2)
  ```

#### 3. **reduction(op:var)**
- **Purpose:** Performs a reduction operation (like sum, min, max) across all threads for the specified variable.
- **Usage:**
  ```cpp
  #pragma omp parallel for reduction(+:sum)
  ```
- **Effect:**  
  - When you use `reduction(op:var)`, OpenMP creates a private copy of `var` for each thread. Each thread works with its own copy during the loop.
  - Each thread updates its private copy using the specified operation (`op`). For example, with `reduction(+:sum)`, each thread adds to its own sum.
  - At the end of the parallel region, OpenMP automatically enters a synchronization phase. This is done in a thread-safe way, typically using a hidden critical section or efficient reduction algorithm, so only one thread updates the shared variable at a time.
  - **Supported Operations:** `+`, `-`, `*`, `&`, `|`, `^`, `&&`, `||`, `min`, `max`
---

### Example with Multiple Clauses

```cpp
int sum = 0;
#pragma omp parallel for collapse(2) reduction(+:sum) schedule(dynamic, 2)
for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
        sum += arr[i][j];
    }
}
```
**Explanation:**
- `collapse(2)`: Parallelizes both `i` and `j` loops together.
- `reduction(+:sum)`: Each thread computes a local sum, then OpenMP combines them.
- `schedule(dynamic, 2)`: Threads are assigned 2 iterations at a time, dynamically.

---
### Summary Table

| Clause/Usage                      | Purpose                                                      |
|-----------------------------------|--------------------------------------------------------------|
| `collapse(n)`                     | Parallelize `n` nested loops as a single loop                |
| `schedule(type[,chunk])`          | Control iteration distribution among threads                 |
| `private(var)`                    | Each thread gets its own copy of `var`                       |
| `shared(var)`                     | All threads share the same `var`                             |
| `reduction(op:var)`               | Combine results from all threads using the specified operator |
---


## The `barrier` Directive

### Purpose

The `barrier` directive is used to synchronize all threads in a parallel region. When a thread reaches a `barrier`, it waits until all other threads in the team have also reached the barrier before any can proceed. This ensures that all threads have completed their work up to that point.

### When to Use

- When you need to make sure all threads have finished a certain section of code before moving on.
- To coordinate phases of computation where results from all threads are needed before proceeding.
- When you want to prevent race conditions that could occur if some threads move ahead too early.

---

### Detailed Example: Without and With Barrier

#### Without Barrier

```cpp
#pragma omp parallel num_threads(4)
{
    int thread_id = omp_get_thread_num();
    printf("Thread %d: Step 1\n", thread_id);

    // No barrier here

    printf("Thread %d: Step 2\n", thread_id);
}
```

**Possible Output (order may vary):**
```
Thread 0: Step 1
Thread 0: Step 2
Thread 3: Step 1
Thread 3: Step 2
Thread 1: Step 1
Thread 1: Step 2
Thread 2: Step 1
Thread 2: Step 2
```
Threads can move to Step 2 as soon as they finish Step 1, even if others are still on Step 1.

---

#### With Barrier

```cpp
#pragma omp parallel num_threads(4)
{
    int thread_id = omp_get_thread_num();
    printf("Thread %d: Step 1\n", thread_id);

    #pragma omp barrier  // All threads must reach here before any continue

    printf("Thread %d: Step 2\n", thread_id);
}
```

Output (Step 2 only starts after all Step 1 are done):
```
Thread 0: Step 1
Thread 1: Step 1
Thread 2: Step 1
Thread 3: Step 1
Thread 1: Step 2
Thread 2: Step 2
Thread 0: Step 2
Thread 3: Step 2
```
Here, no thread prints "Step 2" until all threads have printed "Step 1".

---

### Why is this useful?

Suppose each thread is filling part of an array in Step 1, and in Step 2, you want to process the whole array. The barrier ensures all threads finish filling before any thread starts processing.

---

### Explanation

- All threads execute code up to the `barrier`.
- Each thread waits at the barrier until every thread in the team has arrived.
- After all threads reach the barrier, they all continue execution past the barrier.

### Notes

- Implicit barriers exist at the end of parallel regions and some other OpenMP constructs (like `for` and `sections` by default).
- You can use the `nowait` clause with some directives to remove the implicit barrier if synchronization is not needed.
### Summary Table

| Directive                | Purpose                                      |
|--------------------------|----------------------------------------------|
| `#pragma omp barrier`    | Synchronize all threads at this point        |

---

# Real Problem
