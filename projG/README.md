# Project G - CUDA

You will redo your Project 1F in CUDA.


## Overview

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

## Description
Your `main()` method will consist of the following:

1. Read Triangles and allocate an Image,
2. For every frame:
    1. Set up camera,
    2. Get lighting parameters based on camera
    3. [GPU] Run Phong shading
    4. [GPU] Transform coordinates from world space to device space (device being screen, not to be confused with GPU device)
    5. [GPU] Rasterize,
    6. Copy image from GPU to CPU memory and save as PNM.
        * *NOTE:* Please make this a single function, because in order to measure rasterization speed, we will have to comment
           this out.

## Kernels
This is a list of kernels you **have** to write.
Obviously you should feel free to write more if you need, but this is just the minimum.

### Shading

This is the recommended format for your shading kernel:

```Cuda
__global__
void phong_shading_kernel(
    // These are your kernel inputs.
    const double * vertex_positions, // Pointer to --all-- vertices.
    const double * vertex_normals,   // Pointer to --all-- normals.
    const double * light_direction,  // Pointer to the light direction. 
                                     // (3D vector; array of size 3) 
    const double * camera_position,  // Pointer to the camera position. 
                                     // (3D vector; array of size 3) 
    // This is what your kernel will compute, therefore not a const.
    double * shading_values,         // Pointer to --all-- shading values
    // Other arguments
    const int num_vertices,
    const double Ka,
    const double Kd,
    const double Ks,
    const double alpha);
```

Instructions should compute a single shading value (double).

### Coordinate Transformation

This is the recommended format for your transformation kernel:

```Cuda
__global__
void transformation_kernel(
    // These are your kernel inputs.
    const double * vertex_positions, // Pointer to --all-- vertices.
    const double * mvp,              // Pointer to the transformation matrix
                                     // (4x4 matrix; array of size 16)
    // This is what your kernel will compute, therefore not a const.
    double * out_vertex_positions,   // Pointer to --all-- transformed vertices.
    // Other arguments
    const int num_vertices);
```

Instructions should compute a single output vertex (3D coordinate; 3 values).

### Rasterization

This is the recommended format for your rasterization kernel.

```Cuda
__global__
void rasterization_kernel(
    // These are your kernel inputs.
    const double * vertex_positions, // Pointer to --all-- vertices (transformed).
                                     // Computed by your second kernel.
    const double * vertex_colors,    // Pointer to --all-- vertex colors. 
    const double * vertex_shadings,  // Pointer to --all-- shading values.
                                     // Computed by your first kernel.
    // This is what your kernel will compute, therefore not a const.
    unsigned char * image_data,      // Pointer to your image array.
                                     // (height x width x 3 array of unsigned chars)
    // This is what your kernel will update.
    double * z_buffer,               // Pointer to your z-buffer array.
                                     // (height x width array of doubles)
    // Other arguments
    const int height,
    const int width,
    const int num_triangles);
```

Instructions should run the scanline algorithm on a single triangle.


## Rubric
You will get full credit as long as your code is readable, 
and your program runs faster than a CPU implementation.

Unfortunately, it's hard to measure that and set a target latency or frames per second, given that people are surely going to
use different cards.

As a starting point, aim for correctness (operations mentioned should run on the GPU),
and you can always improve from there.

## Structure

The skeleton code is comprised of the following:

### Model struct
This is basically a replacement for your `TriangleList`. 
Instead of having an array of `Triangle` instances, where each instance has 3 vertices (3D), 3 normals (3D), 3 colors (RGB), and
a shading value (computed by doing Phong shading), you have a single instance holding all the vertices, all the normals, all the
colors, and all the shading values.
This is easier for a number of reasons, including, but not limited to easier transformation and preprocessing, and simplified
kernel launches.

```Cuda
struct Model {
  // The following are read from the file and only once
  int numTriangles;
  double * vertices;     // Original coordinates read from file
  double * normals;      // Normals read from file 
  double * colors;       // Colors read from file
  
  // The following are computed prior to every rasterization
  double * out_vertices; // Will hold transformed vertex coordinated
  double * shading;      // Will hold shading values
};
```

You're also given a modified version of the triangle reader that came with the 1E and 1F starter code, which reads the same 
`ws_tris.txt` file, but stores it in a `Model` instance, and it directly stores these values in the GPU.

```Cuda
Model ReadTriangles();
// Reads `ws_tris.txt` from Project 1E/1F,
// Returns an instance of Model with `numTriangles`, 
// `vertices`, `normals`, and `colors` filled.
// Note that the three arrays are on the GPU (device).
```

## CUDA methods you need to know
If you haven't had any exposure to CUDA, please make sure you read this section before starting.

### Macros
We already know that in the context of CUDA, **device** refers to the GPU, and **host** refers to the CPU.
To differentiate between what runs on device and what runs on host, we need to use decorators:

```Cuda
__global__ // Wrap your CUDA kernels with this
__device__ // Wrap your CUDA functions (called from other device code) with this
__host__   // CPU function (optional)
```

For example, if I'm writing a **kernel** that computes a matrix multiplication, I would do:
```Cuda
__global__
void matrix_multiply (...) {
  // Blocks (thread blocks) are arrays of threads. 
  // You specify how many thread blocks you launch, 
  // and how many threads per thread block when you 
  // launch your kernel.
  // 
  // CUDA lets you access the block and thread index 
  // within the kernel, so that you can specify 
  // instructions for every thread (i.e. 
  // calculate phong shading for which vertex, 
  // which vertex to transform,
  // which triangle to rasterize, 
  // etc.) 
  //
  // The "default" pattern you would see is having
  // one "linear" index, which is just an index from
  // 0 through N, N being the total number of threads
  // in all thread blocks. Then you can optionally map
  // your linear index to an index in structures such
  // as triangles, matrices, tensors, etc.
  
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = index / num_columns;
  const int j = index % num_columns;
  
  // Calculate the cell in the i-th row and j-th column of the output matrix.
}
```

However, if I already have a kernel that needs to call a matrix multiplication, I need to write a function.
This means that every single thread can do its own matrix multiplication:
```Cuda
__device__
double * matrix_multiply_fn(...) {
  // Calculate an entire matrix multiplication.
}
```

Your device functions are more similar to normal CPU functions in that there's no parallelization. There's one core running all
of those instructions. 
However, kernels are "launched" (when you pass in the number of blocks and threads per block using `<<<blocks, threads>>>`), 
and once they are you're essentially calling the kernel function multiple times in parallel.

### Atomic operations
As discussed in the CUDA lecture, atomic operations are used to handle **race conditions**, which is when more than one thread is
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

```Cuda
gpuAtomicMin(double * address, double value);
gpuAtomicMin(float  * address, float  value);

gpuAtomicMax(double * address, double value);
gpuAtomicMax(float  * address, float  value);
```

implemented for you!
Use them wisely (Atomics are obviously more expensive, and not always deterministic. That said, you're okay using them to
implement z-buffer and still get decent performance with no obvious issues in the output.)

### Synchronization
CUDA kernel launches are asynchronous. 
This means that once the CPU ( *host* ) launches a kernel, the GPU ( *device* ) will start scheduling threads and executing the
kernel, and the CPU will continue to run through its own instructions.
This means that if your kernel launch is followed by a CPU-only instruction, it will be executed possibly before your kernel
launch is concluded.

Because of that, it is good practice to do a `cudaDeviceSynchronize();` after kernel launches:

```Cuda
rasterization_kernel<<<BLOCKS, THREADS>>>(...);
cudaDeviceSynchronize();
```

### Error checking
CUDA does not throw errors normally. 
You could launch a kernel and have it fail and you wouldn't notice it until you look at the output.
The standard function to check for CUDA errors is `cudaGetLastError`, which returns an enum error type.
To make things easier, we borrowed a function that gets the last error, and kills your program if there's a failure, and
even points you at the line of code (where you called it):

```Cuda
CHECK_LAST_CUDA_ERROR();
```

When in doubt, add this line next to a kernel launch or any CUDA functions to see if CUDA threw any errors.

### Memory allocation
To allocate memory on the GPU ( *device* ), you will need to use `cudaMalloc`, which is similar to `malloc`:

```Cuda
double * array_gpu;
cudaMalloc(&array_gpu, sizeof(double) * array_size);
```

Always, always, always free memory after you're done with it.
Allocated device memory will not be freed if you leave the scope.
If you don't free memory, you might end up with out of memory or illegal memory access errors from CUDA.

```Cuda
cudaFree(array_gpu);
```


## Common errors

### Segmentation fault
If you come across segmentation faults, you're more than likely giving the CPU an address to CUDA memory!
This means that you probably tried to access or write CUDA memory from host code.
If this ever happens, make sure to double check that you're moving every device value to host with `cudaMemcpy` before accessing
it:

```Cuda
// Copy from main memory to CUDA memory
// Notice how the CPU pointer is the SECOND argument
cudaMemCpy(pointer_gpu, pointer_cpu, size, cudaMemcpyHostToDevice);

// Copy from CUDA memory to main memory
// Notice how the CPU pointer is the FIRST argument
cudaMemCpy(pointer_cpu, pointer_gpu, size, cudaMemcpyDeviceToHost);
```

### [CUDA] Launch failed
Your kernel launch might fail due to a number of reasons, and you typically get a reason that you can look up.

If the failure was due to "too many resources" being requested, you are either requesting too many threads per threadblock
(maximum should always be 1024, but we strongly encourage you to start with 128 and at maximum use 512 to be safe.),
or the number of threads you're requesting use up more than the available resources per SM (i.e. too many registers.)

Assuming you face this issue, reduce the number of threads per threadblock and succeed, finish your work and look into cutting
down on such resources (i.e. use fewer variables in your kernel, and the like) when you're done, and then increase your 
threads per threadblock.


## Authors
This project was contributed to CS 441 by [Ali Hassani](https://alihassanijr.com).
Feel free to reach out to me if you have any questions: [alih@uoregon.edu](mailto:alih@uoregon.edu).

CS 441/541 (Winter 2023) was instructed by [Prof. Hank Childs](https://cdux.cs.uoregon.edu/childs.html).
