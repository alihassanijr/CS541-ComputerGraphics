/*

CS 441/541

Project G - CUDA

*/
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>
#define DEVICE __device__ inline
#define HOST __host__
#define KERNEL __global__

#define INDEX3(i, j) i * 3 + j
#define INDEX4(i, j) i * 4 + j

#define N_FRAMES 50

#define HEIGHT 1000
#define WIDTH  1000

#define FILL_NUM_THREADS       128
#define SHADER_NUM_THREADS     256
#define MVP_NUM_THREADS        128
#define RASTERIZER_NUM_THREADS 256

#define DATATYPE double

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
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
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
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Imported from Ali's 1F
/// Value swap
template <typename T>
DEVICE
void swap(T* a, T* b) {
  T tmp = a[0];
  a[0] = b[0];
  b[0] = tmp;
}

/// CS441 Ceil function.
DEVICE
DATATYPE C441(DATATYPE f) {
  return ceil(f-0.00001);
}

/// CS441 Floor function.
DEVICE
DATATYPE F441(DATATYPE f) {
  return floor(f+0.00001);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////

// Model struct will hold vertices as a single matrix, transformed vertices as another, one for normals
// one for colors, and one for shader values.
struct Model {
  int numTriangles;
  DATATYPE * vertices;
  DATATYPE * out_vertices;
  DATATYPE * normals;
  DATATYPE * colors;
  DATATYPE * shading;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Borrowed from the 1E and 1F starter code
// Functions that read the aneurysm 3D model from the text file
// Every triangle read is directly copied into CUDA.
char * Read3Numbers(char *tmp, DATATYPE *v1, DATATYPE *v2, DATATYPE *v3) {
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
  cudaMalloc(&m.vertices, sizeof(DATATYPE) * numTriangles * 9);
  cudaMalloc(&m.normals,  sizeof(DATATYPE) * numTriangles * 9);
  cudaMalloc(&m.colors,   sizeof(DATATYPE) * numTriangles * 9);
  
  for (int i = 0 ; i < numTriangles ; i++) {
    DATATYPE coords[9];
    DATATYPE colors[9];
    DATATYPE normals[9];
    for (int j = 0 ; j < 3 ; j++) {
      tmp = Read3Numbers(tmp, coords + (j*3), coords + (j*3+1), coords + (j*3+2));
      tmp += 3; /* space+slash+space */
      tmp = Read3Numbers(tmp, colors + (j*3), colors + (j*3+1), colors + (j*3+2));
      tmp += 3; /* space+slash+space */
      tmp = Read3Numbers(tmp, normals + (j*3), normals + (j*3+1), normals + (j*3+2));
      tmp++;    /* newline */
    }
    cudaMemcpy(m.vertices + (i * 9),  coords, 9 * sizeof(DATATYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(m.colors   + (i * 9),  colors, 9 * sizeof(DATATYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(m.normals  + (i * 9), normals, 9 * sizeof(DATATYPE), cudaMemcpyHostToDevice);
  }
  
  free(buffer);
  return m;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copied from 1F but slightly modified
struct Camera {
  DATATYPE near, far;
  DATATYPE angle;
  DATATYPE* position;
  DATATYPE* focus;
  DATATYPE* up;

  Camera (DATATYPE* position, DATATYPE* focus, DATATYPE* up, DATATYPE angle, DATATYPE near, DATATYPE far) :
    position(position),
    focus(focus),
    up(up),
    angle(angle),
    near(near),
    far(far) {}
};

struct LightingParameters {
  DATATYPE * lightDir;   // The direction of the light source
  DATATYPE Ka;           // The coefficient for ambient lighting.
  DATATYPE Kd;           // The coefficient for diffuse lighting.
  DATATYPE Ks;           // The coefficient for specular lighting.
  DATATYPE alpha;        // The exponent term for specular lighting.

  LightingParameters (DATATYPE* lightDir, DATATYPE Ka, DATATYPE Kd, DATATYPE Ks, DATATYPE alpha) :
    lightDir(lightDir),
    Ka(Ka),
    Kd(Kd),
    Ks(Ks),
    alpha(alpha) {}
};

// Cotangent
template <typename T>
DEVICE
T cot(T v) {
  return T(1.0) / tan(v);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Device functions and kernels
DEVICE
DATATYPE device_dot_product(const DATATYPE * a, const DATATYPE * b, const int n) {
  DATATYPE dp = 0.0;
  #pragma unroll
  for (int i=0; i < n; ++i)
    dp += a[i] * b[i];
  return dp;
}

KERNEL
void cross3(const DATATYPE* a, const DATATYPE* b, DATATYPE* c) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex == 0) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2]; // -1 already applied
    c[2] = a[0] * b[1] - a[1] * b[0];
  }
}

KERNEL
void sum(const DATATYPE * A, DATATYPE * sum, const int size) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex == 0) {
    #pragma unroll
    for (int i = 0; i < size; ++ i) {
      sum[0] += A[i];
    }
  }
}

KERNEL
void l2_norm(const DATATYPE * A, DATATYPE * norm, const int size) {
  // norm = ||A||_2
  // A \in R^{size}
  // norm \in R
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex == 0) {
    #pragma unroll
    for (int i = 0; i < size; ++ i) {
      norm[0] += pow(A[i], 2);
    }
    norm[0] = sqrt(norm[0]);
  }
}

KERNEL
void elementwise_prod(const DATATYPE * A, const DATATYPE * B, DATATYPE * D, const int size) {
  // D = A . B
  // D, A, B \in R^{size}
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex < size) {
    D[linearIndex] = A[linearIndex] * B[linearIndex];
  }
}

KERNEL
void elementwise_subtract(const DATATYPE * A, const DATATYPE * B, DATATYPE * D, const int size) {
  // D = A + B
  // D, A, B \in R^{size}
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex < size) {
    D[linearIndex] = A[linearIndex] - B[linearIndex];
  }
}

KERNEL
void elementwise_add(const DATATYPE * A, const DATATYPE * B, DATATYPE * D, const int size) {
  // D = A + B
  // D, A, B \in R^{size}
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex < size) {
    D[linearIndex] = A[linearIndex] + B[linearIndex];
  }
}

KERNEL
void scalar_prod(const DATATYPE * A, const DATATYPE * s, DATATYPE * D, const int size) {
  // D = A . s
  // D, A \in R^{size}
  // s \in R
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex < size) {
    D[linearIndex] = A[linearIndex] * s[0];
  }
}

KERNEL
void scalar_prod(const DATATYPE * A, const DATATYPE s, DATATYPE * D, const int size) {
  // D = A . s
  // D, A \in R^{size}
  // s \in R
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex < size) {
    D[linearIndex] = A[linearIndex] * s;
  }
}

KERNEL
void scalar_div(const DATATYPE * A, const DATATYPE * s, DATATYPE * D, const int size) {
  // D = A / s
  // D, A \in R^{size}
  // s \in R
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex < size && s != 0) {
    D[linearIndex] = A[linearIndex] / s[0];
  }
}

KERNEL
void scalar_div(const DATATYPE * A, const DATATYPE s, DATATYPE * D, const int size) {
  // D = A / s
  // D, A \in R^{size}
  // s \in R
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex < size && s != 0) {
    D[linearIndex] = A[linearIndex] / s;
  }
}

KERNEL
void matmul(const DATATYPE * A, const DATATYPE * B, DATATYPE * C, const int m, const int n, const int k) {
  // C = A B
  // A \in R^{m x k}
  // B \in R^{k x n}
  // C \in R^{m x n}
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearIndex < m * n) {
    const int y = linearIndex % n;
    const int x = linearIndex / n;

    C[linearIndex] = 0.0;

    #pragma unroll
    for (int z=0; z < k; ++z) {
      C[linearIndex] += A[x * k + z] * B[z * n + y]; 
    }
  }
}

template <typename T = DATATYPE>
KERNEL
void fill(T * data, const T val, const int size) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex < size) {
    data[linearIndex] = val;
  }
}

template <typename T = DATATYPE>
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

template <typename T = DATATYPE>
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

template <typename T = DATATYPE>
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

template <typename T = DATATYPE>
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

template <typename T = DATATYPE>
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

HOST
DATATYPE* dot_product(const DATATYPE* a, const DATATYPE* b, const int size) {
  DATATYPE* d = zeros_cuda(1, size);
  DATATYPE* dot_prod = zeros_cuda(1, 1);

  // Hadamard product kernel call
  int max_threads = 4;
  int blocks = (size + max_threads - 1) / max_threads;
  elementwise_prod<<<blocks, max_threads>>>(a, b, d, size);
  //cudaDeviceSynchronize(); 

  // Sum kernel call
  sum<<<1, 1>>>(d, dot_prod, size);
  //cudaDeviceSynchronize(); 
  return dot_prod;
}

HOST
DATATYPE* normalize(const DATATYPE* vec, const int size) {
  DATATYPE* norm = zeros_cuda(1, 1);
  DATATYPE* vec_out = zeros_cuda(1, size);

  // Calculate norm with no parallelism
  // We only do it this way because it's not worth the parallelism in our typical use case (size <= 4)
  l2_norm<<<1, 1>>>(/* vector = */ vec, norm, /* size = */ size);
  //cudaDeviceSynchronize(); 

  // Normalize kernel call
  int max_threads = 4;
  int blocks = (size + max_threads - 1) / max_threads;
  scalar_div<<<blocks, max_threads>>>(vec, norm, vec_out, size);
  //cudaDeviceSynchronize(); 
  return vec_out;
}

KERNEL
void device_transform_kernel(DATATYPE * mat, const DATATYPE height, const DATATYPE width) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearIndex == 0) {
    DATATYPE scale_h = height / 2;
    DATATYPE scale_w = width / 2;
    mat[INDEX4(0, 0)] = scale_w;
    mat[INDEX4(1, 1)] = scale_h;
    mat[INDEX4(2, 2)] = 1.0;
    mat[INDEX4(3, 0)] = scale_w;
    mat[INDEX4(3, 1)] = scale_h;
    mat[INDEX4(3, 3)] = 1.0;
  }
}

KERNEL
void camera_transform_kernel(DATATYPE * mat, const DATATYPE * u_, const DATATYPE * v_, const DATATYPE * w_,
    const DATATYPE * ut, const DATATYPE * vt, const DATATYPE * wt) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearIndex == 0) {
    mat[INDEX4(0, 0)] = u_[0];
    mat[INDEX4(0, 1)] = v_[0];
    mat[INDEX4(0, 2)] = w_[0];
    mat[INDEX4(1, 0)] = u_[1];
    mat[INDEX4(1, 1)] = v_[1];
    mat[INDEX4(1, 2)] = w_[1];
    mat[INDEX4(2, 0)] = u_[2];
    mat[INDEX4(2, 1)] = v_[2];
    mat[INDEX4(2, 2)] = w_[2];

    mat[INDEX4(3, 0)] = ut[0];
    mat[INDEX4(3, 1)] = vt[0];
    mat[INDEX4(3, 2)] = wt[0];
    mat[INDEX4(3, 3)] = 1.0;
  }
}

KERNEL
void view_transform_kernel(DATATYPE * mat, const DATATYPE angle, const DATATYPE near, const DATATYPE far) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearIndex == 0) {
    DATATYPE cot_alpha_div_2 = cot(angle / 2.0);
    mat[INDEX4(0, 0)] = cot_alpha_div_2;
    mat[INDEX4(1, 1)] = cot_alpha_div_2;
    mat[INDEX4(2, 2)] = (far + near) / (far - near);
    mat[INDEX4(3, 2)] = (2 * far * near) / (far - near);
    mat[INDEX4(2, 3)] = -1.0;
  }
}

DATATYPE* device_transform(const DATATYPE height, const DATATYPE width) {
  DATATYPE* mat = zeros_cuda(4, 4);
  device_transform_kernel<<<1, 1>>>(mat, height, width);
  //cudaDeviceSynchronize();
  return mat;
}

DATATYPE* camera_transform(Camera camera) {
  DATATYPE* mat = zeros_cuda(4, 4);
  DATATYPE* w_ = zeros_cuda(1, 3);
  DATATYPE* u_ = zeros_cuda(1, 3);
  DATATYPE* v_ = zeros_cuda(1, 3);
  DATATYPE* t_ = zeros_cuda(1, 3);

  // Compute w
  elementwise_subtract<<<1, 3>>>(/* A = */ camera.position, /* B = */ camera.focus, /* out = */ w_, /* size = */ 3);
  //cudaDeviceSynchronize(); 
  w_ = normalize(/* vec = */ w_, /* size = */ 3);

  // Compute u
  cross3<<<1, 1>>>(camera.up, w_, /* out = */ u_);
  //cudaDeviceSynchronize(); 
  u_ = normalize(/* vec = */ u_, /* size = */ 3);

  // Compute v
  cross3<<<1, 1>>>(w_, u_, /* out = */ v_);
  //cudaDeviceSynchronize(); 
  // No need to normalize v because it's the cross product of two unit vectors

  // Compute t
  scalar_prod<<<1, 3>>>(camera.position, -1.0, t_, 3);
  //cudaDeviceSynchronize(); 

  // Compute u t
  DATATYPE* ut = dot_product(u_, t_, 3);
  DATATYPE* vt = dot_product(v_, t_, 3);
  DATATYPE* wt = dot_product(w_, t_, 3);

  // Arrange values
  camera_transform_kernel<<<1, 1>>>(mat, u_, v_, w_, ut, vt, wt);
  //cudaDeviceSynchronize(); 

  return mat;
}

DATATYPE* view_transform(Camera camera) {
  DATATYPE* mat = zeros_cuda(4, 4);
  view_transform_kernel<<<1, 1>>>(mat, camera.angle, camera.near, camera.far);
  //cudaDeviceSynchronize(); 
  return mat;
}

HOST
DATATYPE SineParameterize(int curFrame, int nFrames, int ramp)
{
    int nNonRamp = nFrames-2*ramp;
    DATATYPE height = 1./(nNonRamp + 4*ramp/M_PI);
    if (curFrame < ramp)
    {
        DATATYPE factor = 2*height*ramp/M_PI;
        DATATYPE eval = cos(M_PI/2*((DATATYPE)curFrame)/ramp);
        return (1.-eval)*factor;
    }
    else if (curFrame > nFrames-ramp)
    {
        int amount_left = nFrames-curFrame;
        DATATYPE factor = 2*height*ramp/M_PI;
        DATATYPE eval =cos(M_PI/2*((DATATYPE)amount_left/ramp));
        return 1. - (1-eval)*factor;
    }
    DATATYPE amount_in_quad = ((DATATYPE)curFrame-ramp);
    DATATYPE quad_part = amount_in_quad*height;
    DATATYPE curve_part = height*(2*ramp)/M_PI;
    return quad_part+curve_part;
}

template <typename T = DATATYPE>
HOST
T* to_cpu(T * data_cuda, const int size, const bool free_src = false) {
  T * data_cpu = (T *)malloc(sizeof(T) * size);
  cudaMemcpy(data_cpu, data_cuda, size * sizeof(T), cudaMemcpyDeviceToHost);
  if (free_src)
    cudaFree(data_cuda);
  return data_cpu;
}

HOST
DATATYPE* to_cuda(DATATYPE * data_cpu, const int size, const bool free_src = false) {
  DATATYPE *data_cuda;
  cudaMalloc(&data_cuda, sizeof(DATATYPE) * size);
  cudaMemcpy(data_cuda, data_cpu, size * sizeof(DATATYPE), cudaMemcpyHostToDevice);
  if (free_src)
    free(data_cpu);
  return data_cuda;
}

HOST
Camera GetCamera(int frame, int nframes) {
  DATATYPE t = SineParameterize(frame, nframes, nframes/10);
  DATATYPE* position = zeros_cpu(1, 3);
  DATATYPE* focus = zeros_cpu(1, 3);
  DATATYPE* up = zeros_cpu(1, 3);
  
  position[0] = 40.0*sin(2*M_PI*t);
  position[1] = 40.0*cos(2*M_PI*t);
  position[2] = 40.0;
  // focus is 0, 0, 0 == zeros
  // up is 0, 1, 0
  up[1] = 1.0;
  DATATYPE near = 5.0;
  DATATYPE far = 200.0;
  DATATYPE angle = M_PI/6.0;
  position = to_cuda(position, 3, true);
  focus    = to_cuda(   focus, 3, true);
  up       = to_cuda(      up, 3, true);
  return Camera(position, focus, up, angle, near, far);
}

HOST
LightingParameters GetLighting(Camera c) {
  DATATYPE Ka = 0.3;
  DATATYPE Kd = 0.7;
  DATATYPE Ks = 2.8;
  DATATYPE alpha = 50.5;

  DATATYPE* lightDir = zeros_cuda(1, 3);
  elementwise_subtract<<<1, 3>>>(/* A = */ c.position, /* B = */ c.focus, /* out = */ lightDir, /* size = */ 3);
  //cudaDeviceSynchronize(); 
  lightDir = normalize(/* vec = */ lightDir, /* size = */ 3);

  return LightingParameters(lightDir, Ka, Kd, Ks, alpha);
}

HOST
DATATYPE* GetTransforms(Camera camera, const DATATYPE height, const DATATYPE width) {
  DATATYPE* D = device_transform(height, width);
  DATATYPE* C = camera_transform(camera);
  DATATYPE* V = view_transform(camera);
  DATATYPE* CV = zeros_cuda(4, 4);
  DATATYPE* output = zeros_cuda(4, 4);
  // Perform 2 matrix multiplications
  // Both 4x4 inputs and 4x4 output
  // So we'd need 16 threads, one for each output scalar.
  matmul<<<1, 16>>>(C, V, CV, 4, 4, 4);
  matmul<<<1, 16>>>(CV, D, output, 4, 4, 4);
  //cudaDeviceSynchronize();
  return output;
}

DEVICE
DATATYPE * device_view_direction(const DATATYPE * c, const DATATYPE * v) {
  // We're using malloc and not cudaMalloc in device code.
  DATATYPE * out = (DATATYPE *)(malloc(sizeof(DATATYPE) * 3));
  DATATYPE norm = 0;
  #pragma unroll
  for (int i=0; i < 3; ++i) {
    DATATYPE acc = c[i] - v[i];
    out[i] = acc;
    norm += pow(acc, 2);
  }
  norm = sqrt(norm);
  #pragma unroll
  for (int i=0; i < 3; ++i) {
    out[i] = out[i] / norm;
  }
  return out;
}

DEVICE
DATATYPE * device_elementwise_prod(const DATATYPE * a, const DATATYPE b, const int n) {
  // We're using malloc and not cudaMalloc in device code.
  DATATYPE * out = (DATATYPE *)(malloc(sizeof(DATATYPE) * n));

  #pragma unroll
  for (int i=0; i < n; ++i) {
    out[i] = a[i] * b;
  }
  return out;
}

DEVICE
DATATYPE * device_elementwise_subtract(const DATATYPE * a, const DATATYPE * b, const int n) {
  // We're using malloc and not cudaMalloc in device code.
  DATATYPE * out = (DATATYPE *)(malloc(sizeof(DATATYPE) * n));

  #pragma unroll
  for (int i=0; i < n; ++i) {
    out[i] = a[i] - b[i];
  }
  return out;
}

DEVICE
DATATYPE * device_elementwise_prod_and_subtract(const DATATYPE * a, const DATATYPE * b, const DATATYPE s, const int n) {
  // We're using malloc and not cudaMalloc in device code.
  DATATYPE * out = (DATATYPE *)(malloc(sizeof(DATATYPE) * n));

  #pragma unroll
  for (int i=0; i < n; ++i) {
    out[i] = s * a[i] - b[i];
  }
  return out;
}

KERNEL
void phong_shader(
    const DATATYPE * vertex_positions, 
    const DATATYPE * vertex_normals, 
    const DATATYPE * light_direction,
    const DATATYPE * camera_position,
    DATATYPE * shading_values,
    const int num_vertices,
    const DATATYPE Ka,
    const DATATYPE Kd,
    const DATATYPE Ks,
    const DATATYPE alpha) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearIndex < num_vertices) {
    DATATYPE LN = device_dot_product(light_direction, vertex_normals + linearIndex * 3, 3);
    DATATYPE diffuse = max(0.0, LN);

    DATATYPE * view_direction = device_view_direction(camera_position, vertex_positions + linearIndex * 3);

    DATATYPE * A = device_elementwise_prod(vertex_normals + linearIndex * 3, (2*LN), 3);
    DATATYPE * R = device_elementwise_subtract(A, light_direction, 3);
    DATATYPE RV = max(0.0, device_dot_product(R, view_direction, 3));
    free(A);
    free(R);
    free(view_direction);

    DATATYPE specular = pow(RV, alpha);
    shading_values[linearIndex] = Ka + Kd * diffuse + Ks * specular;
  }
}

KERNEL
void mvp_transform(
    const DATATYPE * vertex_positions,
    const DATATYPE * mvp,
    DATATYPE * out_vertex_positions,
    const int num_vertices) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = linearIndex / 3;
  const int j = linearIndex % 3;
  if (i < num_vertices) {
    DATATYPE w = mvp[3 * 4 + 3]; // 4th dimension
    DATATYPE accum = mvp[3 * 4 + j]; // 1.0 column
  #pragma unroll
    for (int k=0; k < 3; ++k) {
      accum += vertex_positions[i * 3 + k] * mvp[k * 4 + j];
      w += vertex_positions[i * 3 + k] * mvp[k * 4 + 3];
    }
    out_vertex_positions[i * 3 + j] = accum / w;
  }
}

HOST
void shader_and_transform(Model *m, Camera c, LightingParameters lp, DATATYPE height, DATATYPE width) {
  DATATYPE* mvp = GetTransforms(c, height, width);
  int num_vertices = m->numTriangles * 3;
  int problem_size = m->numTriangles * 9;

  cudaMalloc(&m->shading, sizeof(DATATYPE) * num_vertices);

  int blocks = (problem_size + SHADER_NUM_THREADS - 1) / SHADER_NUM_THREADS;
  int threads = SHADER_NUM_THREADS;
  phong_shader<<<blocks, threads>>>(
      m->vertices, m->normals, 
      lp.lightDir, c.position, 
      m->shading,
      num_vertices,
      lp.Ka,
      lp.Kd,
      lp.Ks,
      lp.alpha);

  cudaMalloc(&m->out_vertices, sizeof(DATATYPE) * problem_size);

  blocks = (problem_size + MVP_NUM_THREADS - 1) / MVP_NUM_THREADS;
  threads = MVP_NUM_THREADS;
  mvp_transform<<<blocks, threads>>>(m->vertices, mvp, m->out_vertices, num_vertices);
  //cudaDeviceSynchronize();
}

struct Image {
  unsigned char * data;
  DATATYPE * z_buffer;
  int height, width;
  int hw, hw3;
  bool on_device;

  Image(unsigned char * data, DATATYPE * z_buffer, int height, int width, bool on_device = true):
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
    fill<<<blocks, FILL_NUM_THREADS>>>(z_buffer, std::numeric_limits<DATATYPE>::lowest(), hw);
    //cudaDeviceSynchronize(); 
  }
};

/// Image2PNM: Dumps an Image instance into a PNM file.
void Image2PNM(Image img, std::string fn) {
  FILE *f = fopen(fn.c_str(), "wb");
  assert(f != NULL);
  fprintf(f, "P6\n");
  fprintf(f, "%d %d\n", img.height, img.width);
  fprintf(f, "%d\n", 255);
  fwrite(img.data, /* height x width = */ img.hw, /* size of pixel = */ 3 * sizeof(unsigned char), f);
  fclose(f);
}

HOST 
Image image_cpu(int height, int width) {
  unsigned char * image_data = zeros_cpu<unsigned char>(height, width * 3);
  DATATYPE * z_buffer = array_cpu(height * width, std::numeric_limits<DATATYPE>::lowest());

  return Image(image_data, z_buffer, height, width);
}

HOST 
Image image_cuda(int height, int width) {
  unsigned char * image_data = zeros_cuda<unsigned char>(height * width * 3);
  DATATYPE * z_buffer = array_cuda(height * width, std::numeric_limits<DATATYPE>::lowest());

  return Image(image_data, z_buffer, height, width);
}

HOST
Image image_to_cpu(Image im) {
  unsigned char * image_cpu = to_cpu<unsigned char>(im.data, im.hw3, /* free_source = */ false);
  return Image(image_cpu, nullptr, im.height, im.width, /* on_device = */ false);
}

/// abs_difference: returns the absolute difference of two values.
template <typename T>
DEVICE
T abs_difference(T a, T b) {
  T diff = a - b;
  if (diff < 0)
    return -1 * diff;
  return diff;
}

/// Line
//// Initialized with a slope and bias, and optionally a "left" and "right" coordinate to support cases
//// where slope is 0 or infinity.
struct Line {
  DATATYPE m, b, x_left, x_right;

  DEVICE
  Line(): m(0), b(0), x_left(-1), x_right(-1) {}
  DEVICE
  Line (DATATYPE m, DATATYPE b, DATATYPE x_left, DATATYPE x_right): m(m), b(b), x_left(x_left), x_right(x_right) {}

  DEVICE
  DATATYPE intersect(DATATYPE y) {
    if (m == 0){
      assert(x_left >= 0 && x_right >= 0 && x_left == x_right);
      return x_left;
    }
    return (y - b) / m;
  }

  DEVICE
  DATATYPE leftIntersection(DATATYPE y) {
    return (m == 0) ? (x_left) : ((y - b) / m);
  }

  DEVICE
  DATATYPE rightIntersection(DATATYPE y) {
    return (m == 0) ? (x_right) : ((y - b) / m);
  }

  DEVICE
  bool valid() {
    if (m == 0 && (x_left < 0 || x_right < 0))
      return false;
    return true;
  }
};

DEVICE
Line intercept(const DATATYPE * a, const DATATYPE * b) {
  if (abs_difference(a[0], b[0]) == 0) {
      // Horizontal line -- to prevent zero division, we just return the leftmost and rightmost X coordinates.
      return Line(0, 0, min(a[0], b[0]), max(a[0], b[0]));
  }
  if (abs_difference(a[1], b[1]) == 0) {
      return Line(); // Vertical lines are considered invalid.
  }
  DATATYPE m_ = (b[1] - a[1]) / (b[0] - a[0]);
  DATATYPE b_ = b[1] - (m_ * b[0]);
  return Line(m_, b_, -1, -1);
}

DEVICE
void set_pixel_cuda(unsigned char * data, DATATYPE * z_buffer, const int height, const int width, 
    int j, int i, DATATYPE z, DATATYPE r, DATATYPE g, DATATYPE b) {
  if (i < 0 || j < 0 || i >= height || j >= width)
    return;
  i = height - i;
  const int pixelIndex = i * width + j;
  gpuAtomicMax(z_buffer + pixelIndex, z);
  if (z_buffer[pixelIndex] == z) {
    data[pixelIndex * 3 + 0] = r;
    data[pixelIndex * 3 + 1] = g;
    data[pixelIndex * 3 + 2] = b;
  }
}

#define LERP1D(AX, BX, CX) ((CX - AX) / (BX - AX))
#define LERP(AX, AY, BX, BY, CX, CY) (sqrt(pow(CX - AX, 2) + pow(CY - AY, 2)) / sqrt(pow(BX - AX, 2) + pow(BY - AY, 2)))

DEVICE
void scanline(
    const DATATYPE * position,
    const DATATYPE * color,
    const DATATYPE * shading,
    unsigned char * image_data,
    DATATYPE * z_buffer,
    const int height,
    const int width,
    int anchorIdx,
    int leftIdx,
    int rightIdx,
    int minIdx,
    int maxIdx) {
  const DATATYPE * anchor = position + anchorIdx * 3;
  const DATATYPE * left   = position + leftIdx   * 3;
  const DATATYPE * right  = position + rightIdx  * 3;
  const DATATYPE * anchorColor = color + anchorIdx * 3;
  const DATATYPE *   leftColor = color + leftIdx   * 3;
  const DATATYPE *  rightColor = color + rightIdx  * 3;
  const DATATYPE anchorShading = shading[anchorIdx];
  const DATATYPE   leftShading = shading[leftIdx  ];
  const DATATYPE  rightShading = shading[rightIdx ];
  const DATATYPE rowMin  = position[minIdx * 3 + /* y offset is 1 */ 1];
  const DATATYPE rowMax  = position[maxIdx * 3 + /* y offset is 1 */ 1];

  /* Scanline */
  Line leftEdge  = intercept( left, anchor);
  Line rightEdge = intercept(right, anchor);
  if (leftEdge.valid() && rightEdge.valid()) {
  #pragma unroll
    for (int r=C441(rowMin); r <= F441(rowMax); ++r) {
      DATATYPE leftEnd  =   leftEdge.leftIntersection(r);
      DATATYPE rightEnd = rightEdge.rightIntersection(r);
      DATATYPE t = LERP(left[0], left[1], anchor[0], anchor[1], leftEnd, r);
      DATATYPE leftZ = left[2] * (1 - t) + anchor[2] * t;
      DATATYPE leftShadingX = leftShading * (1 - t) + anchorShading * t;
      DATATYPE leftColorR = leftColor[0] * (1 - t) + anchorColor[0] * t;
      DATATYPE leftColorG = leftColor[1] * (1 - t) + anchorColor[1] * t;
      DATATYPE leftColorB = leftColor[2] * (1 - t) + anchorColor[2] * t;
      t = LERP(right[0], right[1], anchor[0], anchor[1], rightEnd, r);
      DATATYPE rightZ = right[2] * (1 - t) + anchor[2] * t;
      DATATYPE rightShadingX = rightShading * (1 - t) + anchorShading * t;
      DATATYPE rightColorR = rightColor[0] * (1 - t) + anchorColor[0] * t;
      DATATYPE rightColorG = rightColor[1] * (1 - t) + anchorColor[1] * t;
      DATATYPE rightColorB = rightColor[2] * (1 - t) + anchorColor[2] * t;
      if (leftEnd >= rightEnd) {
        swap<DATATYPE>(&leftZ, &rightZ);
        swap<DATATYPE>(&leftColorR, &rightColorR);
        swap<DATATYPE>(&leftColorG, &rightColorG);
        swap<DATATYPE>(&leftColorB, &rightColorB);
        swap<DATATYPE>(&leftShadingX, &rightShadingX);
        swap<DATATYPE>(&leftEnd, &rightEnd);
      }
      #pragma unroll
      for (int c = C441(leftEnd); c <= F441(rightEnd); ++c) {
        DATATYPE tc = LERP1D(leftEnd, rightEnd, c);
        DATATYPE z = leftZ * (1-tc) + rightZ * tc;
        DATATYPE shading = leftShadingX * (1-tc) + rightShadingX * tc;
        DATATYPE color_r = leftColorR * (1-tc) + rightColorR * tc;
        DATATYPE color_g = leftColorG * (1-tc) + rightColorG * tc;
        DATATYPE color_b = leftColorB * (1-tc) + rightColorB * tc;
        set_pixel_cuda(image_data, z_buffer, height, width, c, r, z, 
              C441(255.0 * min(max(0.0, color_r * shading), 1.0)), 
              C441(255.0 * min(max(0.0, color_g * shading), 1.0)), 
              C441(255.0 * min(max(0.0, color_b * shading), 1.0))
              );
      }
    }
  }
}

KERNEL
void rasterization_kernel(
    const DATATYPE * vertex_positions, 
    const DATATYPE * vertex_colors,
    const DATATYPE * vertex_shadings,
    unsigned char * image_data,
    DATATYPE * z_buffer,
    const int height,
    const int width,
    const int num_triangles
    ) {
  const int triangleIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (triangleIdx < num_triangles) {
    const int vertexOffset = triangleIdx * 9;
    const int shadingOffset = triangleIdx * 3;
    const DATATYPE * position = vertex_positions + vertexOffset;
    const DATATYPE * colors   = vertex_colors    + vertexOffset;
    const DATATYPE * shadings = vertex_shadings  + shadingOffset;

    int topv = -1;
    int midv = -1;
    int botv = -1;
    int tlv  = -1;
    int trv = -1;
    int blv  = -1;
    int brv = -1;
    if (position[INDEX3(0, 1)] >= max(position[INDEX3(1, 1)], position[INDEX3(2, 1)])){
      topv = 0;
      if (position[INDEX3(1, 1)] >= position[INDEX3(2, 1)]) {
        midv = 1;
        botv = 2;
      }
      else {
        midv = 2;
        botv = 1;
      }
    }
    else if (position[INDEX3(1, 1)] >= max(position[INDEX3(0, 1)], position[INDEX3(2, 1)])){
      topv = 1;
      if (position[INDEX3(0, 1)] >= position[INDEX3(2, 1)]) {
        midv = 0;
        botv = 2;
      }
      else {
        midv = 2;
        botv = 0;
      }
    }
    else {
      topv = 2;
      if (position[INDEX3(0, 1)] >= position[INDEX3(1, 1)]) {
        midv = 0;
        botv = 1;
      }
      else {
        midv = 1;
        botv = 0;
      }
    }
    if (position[INDEX3(midv, 0)] >= position[INDEX3(botv, 0)]) {
      trv = midv;
      tlv  = botv;
    } else {
      trv = botv;
      tlv  = midv;
    }
    if (position[INDEX3(midv, 0)] >= position[INDEX3(topv, 0)]) {
      brv = midv;
      blv  = topv;
    } else {
      brv = topv;
      blv  = midv;
    }

    scanline(position, colors, shadings, image_data, z_buffer, height, width, botv, blv, brv, /* rowMin = */botv, /* rowMax = */midv);
    scanline(position, colors, shadings, image_data, z_buffer, height, width, topv, tlv, trv, /* rowMin = */midv, /* rowMax = */topv);
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
  //cudaDeviceSynchronize();
}

HOST
std::string gen_filename(int f) {
  char str[256];
  #ifdef VIDEO
  sprintf(str, "frames/projG_frame%04d.pnm", f);
  #else
  sprintf(str, "projG_frame%04d.pnm", f);
  #endif
  return str;
}


int main() {
  Model model = ReadTriangles();
  Image image = image_cuda(HEIGHT, WIDTH);
  #ifdef VIDEO
  for (int f=0; f < N_FRAMES; ++f) {
    #ifdef VERBOSE
    std::cout << "Generating frame " << f << std::endl;
    #endif
  #else
  DATATYPE f = 0;
  #endif
    Camera camera = GetCamera(f, N_FRAMES);
    LightingParameters lp = GetLighting(camera);
    shader_and_transform(&model, camera, lp, HEIGHT, WIDTH);
    //CHECK_LAST_CUDA_ERROR();
    image.clear();
    //CHECK_LAST_CUDA_ERROR();
    rasterize(&image, model);
    //CHECK_LAST_CUDA_ERROR();
    if (image.on_device) {
      Image2PNM(image_to_cpu(image), gen_filename(f));
    } else {
      Image2PNM(image, gen_filename(f));
    }
    CHECK_LAST_CUDA_ERROR();
  #ifdef VIDEO
  }
  #endif
  return 0;
}
