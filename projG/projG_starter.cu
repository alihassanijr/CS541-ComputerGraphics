/*

CS 441/541

Project 3F - CUDA Rasterizer from scratch.

Reach out to alih@uoregon.edu if you have any questions.

*/

// Standard CUDA headers
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>

// These macros just make it easier to define device functions,
// host functions, and kernels.
// The extra `inline` in device is there for possible compile-time 
// optimizations.
#define DEVICE __device__ inline
#define HOST __host__
#define KERNEL __global__

// These macros will help with indexing in matrices that are
// 1D arrays.
// In an Nx3 matrix, you can do:
//   arr[INDEX3(i, j)]
// to get row i col j.
// In an Nx4 matrix, you can do the same by using INDEX4.
#define INDEX3(i, j) i * 3 + j
#define INDEX4(i, j) i * 4 + j

// Image specs
#define N_FRAMES 1000
#define HEIGHT   1000
#define WIDTH    1000

// You can factor out the number of threads per threadblock
// here and change them when you're done to observe throughput
// differences.
#define FILL_NUM_THREADS       128
#define SHADING_NUM_THREADS    256
#define TRANSFORM_THREADS      128
#define RASTERIZER_NUM_THREADS 256

// Other standard C/C++ headers you might need
#include <iostream>
#include <string>
#include <assert.h>
#include <limits>

/// CUDA only implements atomicMin and atomicMax for integers.
/// So we need our own for zbuffer.
/// This is borrowed from PyTorch's ATen backend.
/// Thank you PyTorch!
/// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/Atomic.cuh

template <typename T>
struct AtomicFPOp;

template <>
struct AtomicFPOp<double> {
  template <typename func_t>
  inline __device__ double operator() (double * address, double val, const func_t& func) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, func(val, assumed));
      // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
  }
};

inline __device__ double gpuAtomicMax(double * address, double val) {
  return AtomicFPOp<double>()(address, val,
                              [](double val, unsigned long long int assumed) {
                                return __double_as_longlong(max(val, __longlong_as_double(assumed)));
                              });
}


inline __device__ double gpuAtomicMin(double * address, double val) {
  return AtomicFPOp<double>()(address, val,
                              [](double val, unsigned long long int assumed) {
                                return __double_as_longlong(min(val, __longlong_as_double(assumed)));
                              });
}

// Dont use a templated function for this since the addition function defaults to the CUDA built-in.
inline __device__ float gpuAtomicMax(float * address, float val) {
  unsigned int* address_as_ull = (unsigned int*)address;
  unsigned int old = *address_as_ull;
  unsigned int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __float_as_int(max(val, __int_as_float(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __int_as_float(old);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA error check functions copied from Lei Mao's blog:
// https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
//////////////////////////////////////////////////////////////////////////////////////////////////////////
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We should exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We should exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////

/// CS441 Ceil function.
DEVICE
double C441(double f) {
  return ceil(f-0.00001);
}

/// CS441 Floor function.
DEVICE
double F441(double f) {
  return floor(f+0.00001);
}

// Cotangent device function
template <typename T>
DEVICE
T cot(T v) {
  return T(1.0) / tan(v);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////

// Model struct will hold vertices as a single matrix, transformed vertices as another, one for normals
// one for colors, and one for shading values.
struct Model {
  int numTriangles;
  double * vertices;
  double * out_vertices;
  double * normals;
  double * colors;
  double * shading;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Borrowed from the 1E and 1F starter code
// Functions that read the aneurysm 3D model from the text file
// Every triangle read is directly copied into CUDA.
char * Read3Numbers(char *tmp, double *v1, double *v2, double *v3) {
  *v1 = atof(tmp);
  while (*tmp != ' ')
     tmp++;
  tmp++; /* space */
  *v2 = atof(tmp);
  while (*tmp != ' ')
     tmp++;
  tmp++; /* space */
  *v3 = atof(tmp);
  while (*tmp != ' ' && *tmp != '\n')
     tmp++;
  return tmp;
}

Model ReadTriangles() {
  FILE *f = fopen("ws_tris.txt", "r");
  if (f == NULL) {
    fprintf(stderr, "You must place the ws_tris.txt file in the current directory.\n");
    exit(EXIT_FAILURE);
  }
  fseek(f, 0, SEEK_END);
  int numBytes = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (numBytes != 3892295) {
    fprintf(stderr, "Your ws_tris.txt file is corrupted.  It should be 3892295 bytes, but you have %d.\n", numBytes);
    exit(EXIT_FAILURE);
  }
  
  char *buffer = (char *) malloc(numBytes);
  if (buffer == NULL) {
    fprintf(stderr, "Unable to allocate enough memory to load file.\n");
    exit(EXIT_FAILURE);
  }
  
  fread(buffer, sizeof(char), numBytes, f);
  
  char *tmp = buffer;
  int numTriangles = atoi(tmp);
  while (*tmp != '\n')
    tmp++;
  tmp++;
  
  if (numTriangles != 14702) {
    fprintf(stderr, "Issue with reading file -- can't establish number of triangles.\n");
    exit(EXIT_FAILURE);
  }
  
  Model m;
  m.numTriangles = numTriangles;
  cudaMalloc(&m.vertices, sizeof(double) * numTriangles * 9);
  cudaMalloc(&m.normals,  sizeof(double) * numTriangles * 9);
  cudaMalloc(&m.colors,   sizeof(double) * numTriangles * 9);
  
  for (int i = 0 ; i < numTriangles ; i++) {
    double coords[9];
    double colors[9];
    double normals[9];
    for (int j = 0 ; j < 3 ; j++) {
      tmp = Read3Numbers(tmp, coords + (j*3), coords + (j*3+1), coords + (j*3+2));
      tmp += 3; /* space+slash+space */
      tmp = Read3Numbers(tmp, colors + (j*3), colors + (j*3+1), colors + (j*3+2));
      tmp += 3; /* space+slash+space */
      tmp = Read3Numbers(tmp, normals + (j*3), normals + (j*3+1), normals + (j*3+2));
      tmp++;    /* newline */
    }
    cudaMemcpy(m.vertices + (i * 9),  coords, 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m.colors   + (i * 9),  colors, 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m.normals  + (i * 9), normals, 9 * sizeof(double), cudaMemcpyHostToDevice);
  }
  
  free(buffer);
  return m;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////

// Copied from 1F but slightly modified
struct Camera {
  double near, far;
  double angle;
  double* position;
  double* focus;
  double* up;

  Camera (double* position, double* focus, double* up, double angle, double near, double far) :
    position(position),
    focus(focus),
    up(up),
    angle(angle),
    near(near),
    far(far) {}
};

// Copied from 1F but slightly modified
struct LightingParameters {
  double * lightDir;   // The direction of the light source
  double Ka;           // The coefficient for ambient lighting.
  double Kd;           // The coefficient for diffuse lighting.
  double Ks;           // The coefficient for specular lighting.
  double alpha;        // The exponent term for specular lighting.

  LightingParameters (double* lightDir, double Ka, double Kd, double Ks, double alpha) :
    lightDir(lightDir),
    Ka(Ka),
    Kd(Kd),
    Ks(Ks),
    alpha(alpha) {}
};

// Copied from 1F
HOST
double SineParameterize(int curFrame, int nFrames, int ramp)
{
    int nNonRamp = nFrames-2*ramp;
    double height = 1./(nNonRamp + 4*ramp/M_PI);
    if (curFrame < ramp)
    {
        double factor = 2*height*ramp/M_PI;
        double eval = cos(M_PI/2*((double)curFrame)/ramp);
        return (1.-eval)*factor;
    }
    else if (curFrame > nFrames-ramp)
    {
        int amount_left = nFrames-curFrame;
        double factor = 2*height*ramp/M_PI;
        double eval =cos(M_PI/2*((double)amount_left/ramp));
        return 1. - (1-eval)*factor;
    }
    double amount_in_quad = ((double)curFrame-ramp);
    double quad_part = amount_in_quad*height;
    double curve_part = height*(2*ramp)/M_PI;
    return quad_part+curve_part;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Device functions and kernels

// Fill kernel
// Fills a device array with a specific scalar value
template <typename T = double>
KERNEL
void fill(T * data, const T val, const int size) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex < size) {
    data[linearIndex] = val;
  }
}

// Allocates an array on host (cpu), and fills it with a given value.
template <typename T = double>
HOST
T* array_cpu(const int size, T value) {
  T * mat = (T *)malloc(sizeof(T) * size);

  // Fill 
  #pragma unroll
  for (int i = 0; i < size; ++i) {
    mat[i] = value;
  }

  return mat;
}

// Allocates a matrix on host (cpu) filled with zeros.
template <typename T = double>
HOST
T* zeros_cpu(const int m, const int k) {
  int size = m * k;
  T * mat = (T *)malloc(sizeof(T) * size);

  // Fill 
  #pragma unroll
  for (int i = 0; i < m; ++i) {
    #pragma unroll
    for (int j = 0; j < k; ++j) {
      mat[i * k + j] = 0.0;
    }
  }

  return mat;
}

// Allocates an array on device (gpu), and fills it with a given value.
template <typename T = double>
HOST
T* array_cuda(const int size, T value) {
  T *mat;
  cudaMalloc(&mat, sizeof(T) * size);

  // Fill kernel call
  int blocks = (size + FILL_NUM_THREADS - 1) / FILL_NUM_THREADS;
  fill<T><<<blocks, FILL_NUM_THREADS>>>(mat, value, size);
  //cudaDeviceSynchronize(); 

  return mat;
}

// Allocates an array on device (gpu) filled with zeros.
template <typename T = double>
HOST
T* zeros_cuda(const int size) {
  T *mat;
  cudaMalloc(&mat, sizeof(T) * size);

  // Fill kernel call
  int blocks = (size + FILL_NUM_THREADS - 1) / FILL_NUM_THREADS;
  fill<T><<<blocks, FILL_NUM_THREADS>>>(mat, 0.0, size);
  //cudaDeviceSynchronize(); 

  return mat;
}

// Allocates a matrix on device (gpu) filled with zeros.
template <typename T = double>
HOST
T* zeros_cuda(const int m, const int k) {
  int size = m * k;
  T *mat;
  cudaMalloc(&mat, sizeof(T) * size);

  // Fill kernel call
  int max_threads = 4;
  int blocks = (size + max_threads - 1) / max_threads;
  fill<T><<<blocks, max_threads>>>(mat, 0.0, size);
  //cudaDeviceSynchronize(); 

  return mat;
}

// Copies a device (gpu) array into host (cpu) memory, and optionally
// frees device memory.
template <typename T = double>
HOST
T* to_cpu(T * data_cuda, const int size, const bool free_src = false) {
  T * data_cpu = (T *)malloc(sizeof(T) * size);
  cudaMemcpy(data_cpu, data_cuda, size * sizeof(T), cudaMemcpyDeviceToHost);
  if (free_src)
    cudaFree(data_cuda);
  return data_cpu;
}

// Copies a host (cpu) array into device (gpu) memory, and optionally
// frees host memory.
HOST
double* to_cuda(double * data_cpu, const int size, const bool free_src = false) {
  double *data_cuda;
  cudaMalloc(&data_cuda, sizeof(double) * size);
  cudaMemcpy(data_cuda, data_cpu, size * sizeof(double), cudaMemcpyHostToDevice);
  if (free_src)
    free(data_cpu);
  return data_cuda;
}

// Copied from 1F but adjusted
// Computes camera position, and returns Camera instance
// with position, up, and focus vectors on device (gpu).
HOST
Camera GetCamera(int frame, int nframes) {
  double t = SineParameterize(frame, nframes, nframes/10);
  double* position = zeros_cpu(1, 3);
  double* focus = zeros_cpu(1, 3);
  double* up = zeros_cpu(1, 3);
  
  position[0] = 40.0*sin(2*M_PI*t);
  position[1] = 40.0*cos(2*M_PI*t);
  position[2] = 40.0;
  // focus is 0, 0, 0 == zeros
  // up is 0, 1, 0
  up[1] = 1.0;
  double near = 5.0;
  double far = 200.0;
  double angle = M_PI/6.0;
  position = to_cuda(position, 3, true);
  focus    = to_cuda(   focus, 3, true);
  up       = to_cuda(      up, 3, true);
  return Camera(position, focus, up, angle, near, far);
}

HOST
LightingParameters GetLighting(Camera c) {
  double Ka = 0.3;
  double Kd = 0.7;
  double Ks = 2.8;
  double alpha = 50.5;

  double* lightDir = zeros_cuda(1, 3);
  // TODO: compute lightDir
  // Choices:
  /// 1. Compute on CPU, copy to GPU
  /// 2. Write a kernel computing it on GPU with very few threads.

  return LightingParameters(lightDir, Ka, Kd, Ks, alpha);
}

HOST
double* GetTransforms(Camera camera, const double height, const double width) {
  double* D /* = device_transform(height, width) */;
  double* C /* = camera_transform(camera) */;
  double* V /* = view_transform(camera) */;
  double* output = zeros_cuda(4, 4);
  // Perform 2 matrix multiplications
  // Both 4x4 inputs and 4x4 output
  // TODO: write a matrix multiplication
  return output;
}

KERNEL
void phong_shading_kernel(
    const double * vertex_positions, 
    const double * vertex_normals, 
    const double * light_direction,
    const double * camera_position,
    double * shading_values,
    const int num_vertices,
    const double Ka,
    const double Kd,
    const double Ks,
    const double alpha) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearIndex < num_vertices) {
    // TODO: compute Phong shading value for vertex at linearIndex
  }
}

KERNEL
void transformation_kernel(
    const double * vertex_positions,
    const double * transforms,
    double * out_vertex_positions,
    const int num_vertices) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = linearIndex / 3;
  const int j = linearIndex % 3;
  if (i < num_vertices) {
    // TODO: compute i-th output vertex's j-th dimension
    // Note: You can recompute the 4th dimension because it's needed in the other
    // three (recall what you did with the 4th dimension.)
  }
}

HOST
void shading_and_transform(Model *m, Camera c, LightingParameters lp, double height, double width) {
  double* transforms = GetTransforms(c, height, width);
  int num_vertices = m->numTriangles * 3; // 3 vertices per triangle.
  int problem_size = m->numTriangles * 9; // 3 vertices per triangle, 3D coordinates, 3x3 = 9

  cudaMalloc(&m->shading, sizeof(double) * num_vertices);

  int blocks = (problem_size + SHADING_NUM_THREADS - 1) / SHADING_NUM_THREADS;
  int threads = SHADING_NUM_THREADS;
  phong_shading_kernel<<<blocks, threads>>>(
      m->vertices, m->normals, 
      lp.lightDir, c.position, 
      m->shading,
      num_vertices,
      lp.Ka,
      lp.Kd,
      lp.Ks,
      lp.alpha);

  cudaMalloc(&m->out_vertices, sizeof(double) * problem_size);

  blocks = (problem_size + TRANSFORM_THREADS - 1) / TRANSFORM_THREADS;
  threads = TRANSFORM_THREADS;
  transformation_kernel<<<blocks, threads>>>(m->vertices, transforms, m->out_vertices, num_vertices);
  //cudaDeviceSynchronize();
}

struct Image {
  unsigned char * data;
  double * z_buffer;
  int height, width;
  int hw, hw3;
  bool on_device;

  Image(unsigned char * data, double * z_buffer, int height, int width, bool on_device = true):
    data(data),
    z_buffer(z_buffer),
    height(height),
    width(width),
    hw(height*width),
    hw3(height*width*3),
    on_device(on_device)
  {}

  HOST 
  void clear() {
    clear_pixels();
    clear_zbuffer();
  }

  HOST
  void clear_pixels() {
    // Fill kernel call
    int blocks = (hw3 + FILL_NUM_THREADS - 1) / FILL_NUM_THREADS;
    fill<unsigned char><<<blocks, FILL_NUM_THREADS>>>(data, 0, hw3);
    //cudaDeviceSynchronize(); 
  }

  HOST
  void clear_zbuffer() {
    // Fill kernel call
    int blocks = (hw + FILL_NUM_THREADS - 1) / FILL_NUM_THREADS;
    fill<<<blocks, FILL_NUM_THREADS>>>(z_buffer, std::numeric_limits<double>::lowest(), hw);
    //cudaDeviceSynchronize(); 
  }
};

// Allocates an Image instance on host (cpu)
HOST 
Image image_cpu(int height, int width) {
  unsigned char * image_data = zeros_cpu<unsigned char>(height, width * 3);
  double * z_buffer = array_cpu(height * width, std::numeric_limits<double>::lowest());

  return Image(image_data, z_buffer, height, width);
}

// Allocates an Image instance on device (gpu)
HOST 
Image image_cuda(int height, int width) {
  unsigned char * image_data = zeros_cuda<unsigned char>(height * width * 3);
  double * z_buffer = array_cuda(height * width, std::numeric_limits<double>::lowest());

  return Image(image_data, z_buffer, height, width);
}

// Copies a device Image to host Image
// Ignores z-buffer
// You'll need to do this in order to save your image to file.
HOST
Image image_to_cpu(Image im) {
  unsigned char * image_cpu = to_cpu<unsigned char>(im.data, im.hw3, /* free_source = */ false);
  return Image(image_cpu, nullptr, im.height, im.width, /* on_device = */ false);
}

// Device (GPU) function
// Sets pixel value
DEVICE
void set_pixel_cuda(unsigned char * data, double * z_buffer, const int height, const int width, 
    int j, int i, double z, double r, double g, double b) {
  if (i < 0 || j < 0 || i >= height || j >= width)
    return;
  i = height - i;
  const int pixelIndex = i * width + j;
  // TODO: implement z-buffer with atomics!
  if (/* z-buffer condition met */) {
    data[pixelIndex * 3 + 0] = r;
    data[pixelIndex * 3 + 1] = g;
    data[pixelIndex * 3 + 2] = b;
  }
}

KERNEL
void rasterization_kernel(
    const double * vertex_positions, 
    const double * vertex_colors,
    const double * vertex_shadings,
    unsigned char * image_data,
    double * z_buffer,
    const int height,
    const int width,
    const int num_triangles
    ) {
  const int triangleIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (triangleIdx < num_triangles) {
    // TODO: Rasterize
  }
}

HOST
void rasterize(Image * image, Model model) {
  // To each triangle its own thread
  int problem_size = model.numTriangles;

  int blocks = (problem_size + RASTERIZER_NUM_THREADS - 1) / RASTERIZER_NUM_THREADS;
  int threads = RASTERIZER_NUM_THREADS;
  rasterization_kernel<<<blocks, threads>>>(
      model.out_vertices,
      model.colors,
      model.shading,
      image->data,
      image->z_buffer,
      image->height,
      image->width,
      problem_size);
}

// If you compile with -DVIDEO, it will start generating frames,
// otherwise it will only generate the first frame
int main() {
  Model model = ReadTriangles();
  Image image = image_cuda(HEIGHT, WIDTH);

  #ifdef VIDEO
  for (int f=0; f < N_FRAMES; ++f) {
  #else
  double f = 0;
  #endif
    Camera camera = GetCamera(f, N_FRAMES);
    LightingParameters lp = GetLighting(camera);
    shading_and_transform(&model, camera, lp, HEIGHT, WIDTH);
    CHECK_LAST_CUDA_ERROR();
    image.clear();
    CHECK_LAST_CUDA_ERROR();
    rasterize(&image, model);
    CHECK_LAST_CUDA_ERROR();
    if (image.on_device) {
      // use image_to_cpu to convert your image to CPU
      // DO NOT free Image! You'll need that space for the next frame.
      // Save as PNM
    } else {
      // Save as PNM
    }
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();
    timeit(&_t, "Generated frame " + std::to_string(f), _start, "Total time elapsed: ", f+1, _rasterize_start);
    cudaFree(model.shading);
    cudaFree(model.out_vertices);
  #ifdef VIDEO
  }
  #endif
  return 0;
}
