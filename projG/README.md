# Project G - CUDA

You will redo your Project 1F in CUDA.

## Description

Project 1 was basically writing your own rasterizer in C++.
We wrote a function saving RGB images to PNM files (which we later convert to PNGs), 
then wrote a simplified scanline algorithm,
added in color interpolation and the z-buffer algorithm.
We then extended the rasterizer to factor in phong shading, and transformations (camera, view, device).
At that stage, we ended up moving the camera around to generate frames that built a whole video, which looked like this:

<img src="../assets/outputs/proj1F.png" width="500" />

However, as we all noticed this was all running on the CPU.
It was only in project 2 that we used OpenGL to basically redo all of that, but without writing our own rasterizer.
All we did was either read triangles, or generate our own, which was relatively easy, because we could define shapes out of said
triangles and move them around, apply transformations to them, ultimately generating our own custom-made dog.

OpenGL made us learn a lot, but in this project, we're going to learn a different set of concepts that are (mostly) specific to
NVIDIA GPUs.

NVIDIA introduced CUDA in 2005-2006, a new API and programming model that allowed programmers to take advantage of the GPUs'
massive parallelism without actually rendering graphics. For example, if your GPU can rasterize pixels so quickly, why not use
it to multiply matrices?

## Skeleton code

### Macros
We already know that in the context of CUDA, *device* refers to the GPU, and *host* refers to the CPU.
To differentiate between what runs on device and what runs on host, we need to use decorators:

```
__global__ // CUDA kernel
__device__ // CUDA function
__host__ // CPU function (optional)
```

For example, if I'm writing a *kernel* that computes a matrix multiplication, I would do:
```
__global__
void matrix_multiply (...) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = index / num_columns;
  const int j = index % num_columns;
  
  // Calculate the cell in the i-th row and j-th column of the output matrix.
}
```

However, if I already have a kernel that needs to call a matrix multiplication, I need to write a function.
This means that every single thread can do its own matrix multiplication:
```
__device__
double * matrix_multiply_fn(...) {
  // Calculate an entire matrix multiplication.
}
```

Your device functions are more similar to normal CPU functions in that there's no parallelization. There's one core running all
of those instructions. 
However, kernels are "launched" (when you pass in the number of blocks and threads per block using `<<<>>>`), and once they are
you're technically calling the kernel function multiple times in parallel.

To make all of this easier, you're provided three macros:

```
KERNEL

DEVICE

HOST
```

So you would write `KERNEL void matrix_multiply` instead of `__global__ void matrix_multiply` and so on.

### Atomic operations
As discussed in the CUDA lecture, atomic operations are used to handle *race conditions*, which is when more than one thread is
going to attempt to update the same address as another.

One very easy instance of that is reduction functions (i.e. sum, mean, norm) where multiple input values are accumulated into a 
single output value; if the input values have more than one thread handling them, there will be a race condition, but if you
have one thread for all them, you might end up throwing too much work onto one thread.
(For what it's worth this is a common problem, and atomics and resolving race conditions are only two ways to avoid such issues,
but not the only ones.)

In our C++ rasterizer, we have one very obvious race condition: the z-buffer algorithm.
If you parallelize the rasterizer in such a way that two triangles are handled by different threads (which you'll almost
definitely have to), then you open up the possibility of two threads attempting to write to the same pixel at the same time, and
as a result, they will read and update the z-buffer at the same time, and that's your race condition.

We can avoid that by using the atomic minimum operation (there's also atomic max, atomic add, and atomic multiplication.)
However, one issue is that the standard CUDA libraries do not implement atomic min and max for floats and doubles.
So we'd have to implement our own either using the compare-and-swap operator (CAS), or rely on existing atomics and typecast.

To make things easier, so that both you and I spend less time on such a minor issue, I copied an existing implementation that I
knew I could rely on being up-to-date. 
Therefore, you have:

```
gpuAtomicMin(double * address, double value)
gpuAtomicMin(float  * address, float  value)

gpuAtomicMax(double * address, double value)
gpuAtomicMax(float  * address, float  value)
```

implemented for you!
Use them wisely (Atomics are obviously more expensive, and not always deterministic. That said, you're okay using them to
implement z-buffer and still get decent performance with no obvious issues in the output.)

### Error handling and debugging
There's a couple of things included in the skeleton that makes checking for errors easier.
One is two error checking functions borrowed from [Lei Mao's blog](https://leimao.github.io/blog/Proper-CUDA-Error-Checking/).

#### Finding CUDA errors
When in doubt, add this line next to a kernel launch to see if CUDA threw any errors:

```
CHECK_LAST_CUDA_ERROR();
```

*NOTE:* CUDA does not tend to terminate and throw an error at you if something went wrong, so until you have this one line
at least once after every kernel call, you will not notice CUDA-related error.s

#### Segmentation fault
If you come across segmentation faults, you're more than likely giving the CPU an address to CUDA memory!
This means that you probably tried to access or write CUDA memory from host code.
If this ever happens, make sure to double check that you're moving every device value to host with `cudaMemcpy` before accessing
it:

```
// Copy from main memory to CUDA memory
// Notice how the CPU pointer is the SECOND argument
cudaMemCpy(pointer_gpu, pointer_cpu, size, cudaMemcpyHostToDevice);

// Copy from CUDA memory to main memory
// Notice how the CPU pointer is the FIRST argument
cudaMemCpy(pointer_cpu, pointer_gpu, size, cudaMemcpyDeviceToHost);
```


## Authors
This project was contributed to CS 441 by [Ali Hassani](https://alihassanijr.com).
Feel free to reach out to me if you have any questions: [alih@uoregon.edu](mailto:alih@uoregon.edu).

CS 441/541 (Winter 2023) was instructed by [Prof. Hank Childs](https://cdux.cs.uoregon.edu/childs.html).
