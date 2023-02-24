/*

CS 441/541

Project G - CUDA

*/
#include <cuda.h>
#define DEVICE __inline__ __device__
#define HOST __host__
#define KERNEL __global__

#define INDEX3(i, j) i * 3 + j
#define INDEX4(i, j) i * 4 + j

#include <iostream>
#include <string>
#include <assert.h>

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
double C441(double f) {
  return ceil(f-0.00001);
}

/// CS441 Floor function.
DEVICE
double F441(double f) {
  return floor(f+0.00001);
}

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

inline __device__ double gpuAtomicMin(double * address, double val) {
  return AtomicFPOp<double>()(address, val,
                              [](double val, unsigned long long int assumed) {
                                return __double_as_longlong(min(val, __longlong_as_double(assumed)));
                              });
}


// CUDA error check functions copied from Lei Mao's blog:
// https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
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

struct Model {
  int numTriangles;
  double * vertices;
  double * out_vertices;
  double * normals;
  double * colors;
  double * shading;
  // int * t_to_v;
};

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

  //// Triangle to vertex map
  ////// Every triangle has 3 pointers to its 3 vertices sorted by y-value,
  ////// two pointers to the left and right vertices w.r.t the top vertex,
  ////// and two pointers to the left and right vertices w.r.t the bottom vertex.
  ////// 3 + 2 + = 7.
  //cudaMalloc(&m.t_to_v, sizeof(int) * numTriangles * 7);
  
  for (int i = 0 ; i < numTriangles ; i++) {
    //double x_values[3];
    //double y_values[3];
    double coords[9];
    double colors[9];
    double normals[9];
    for (int j = 0 ; j < 3 ; j++) {
      tmp = Read3Numbers(tmp, coords + (j*3), coords + (j*3+1), coords + (j*3+2));
      //x_values[j] = coords[j*3];
      //y_values[j] = coords[j*3+1];
      tmp += 3; /* space+slash+space */
      tmp = Read3Numbers(tmp, colors + (j*3), colors + (j*3+1), colors + (j*3+2));
      tmp += 3; /* space+slash+space */
      tmp = Read3Numbers(tmp, normals + (j*3), normals + (j*3+1), normals + (j*3+2));
      tmp++;    /* newline */
    }
    cudaMemcpy(m.vertices + (i * 9),  coords, 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m.colors   + (i * 9),  colors, 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m.normals  + (i * 9), normals, 9 * sizeof(double), cudaMemcpyHostToDevice);
    //int * host_t2v = (int *)malloc(sizeof(int) * 7);
    //int top_idx = -1;
    //int mid_idx = -1;
    //int bot_idx = -1;
    //int top_left  = -1;
    //int top_right = -1;
    //int bot_left  = -1;
    //int bot_right = -1;
    //if (y_values[0] >= max(y_values[1], y_values[2])){
    //  top_idx = 0;
    //  if (y_values[1] >= y_values[2]) {
    //    mid_idx = 1;
    //    bot_idx = 2;
    //  }
    //  else {
    //    mid_idx = 2;
    //    bot_idx = 1;
    //  }
    //}
    //else if (y_values[1] >= max(y_values[0], y_values[2])){
    //  top_idx = 1;
    //  if (y_values[0] >= y_values[2]) {
    //    mid_idx = 0;
    //    bot_idx = 2;
    //  }
    //  else {
    //    mid_idx = 2;
    //    bot_idx = 0;
    //  }
    //}
    //else {
    //  top_idx = 2;
    //  if (y_values[0] >= y_values[1]) {
    //    mid_idx = 0;
    //    bot_idx = 1;
    //  }
    //  else {
    //    mid_idx = 1;
    //    bot_idx = 0;
    //  }
    //}
    //if (x_values[mid_idx] >= x_values[bot_idx]) {
    //  top_right = mid_idx;
    //  top_left  = bot_idx;
    //} else {
    //  top_right = bot_idx;
    //  top_left  = mid_idx;
    //}
    //if (x_values[mid_idx] >= x_values[top_idx]) {
    //  bot_right = mid_idx;
    //  bot_left  = top_idx;
    //} else {
    //  bot_right = top_idx;
    //  bot_left  = mid_idx;
    //}
    //host_t2v[0] = top_idx;
    //host_t2v[1] = mid_idx;
    //host_t2v[2] = bot_idx;
    //host_t2v[3] = top_left;
    //host_t2v[4] = top_right;
    //host_t2v[5] = bot_left;
    //host_t2v[6] = bot_right;
    //cudaMemcpy(m.t_to_v + (i * 7), host_t2v, 7 * sizeof(int), cudaMemcpyHostToDevice);
  }
  
  free(buffer);
  return m;
}

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

// Cotangent
template <typename T>
DEVICE
T cot(T v) {
  return T(1.0) / tan(v);
}

DEVICE
double device_dot_product(const double * a, const double * b, const int n) {
  double dp = 0.0;
  for (int i=0; i < n; ++i)
    dp += a[i] * b[i];
  return dp;
}

KERNEL
void cross3(const double* a, const double* b, double* c) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex == 0) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2]; // -1 already applied
    c[2] = a[0] * b[1] - a[1] * b[0];
  }
}

KERNEL
void sum(const double * A, double * sum, const int size) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex == 0) {
    for (int i = 0; i < size; ++ i) {
      sum[0] += A[i];
    }
  }
}

KERNEL
void l2_norm(const double * A, double * norm, const int size) {
  // norm = ||A||_2
  // A \in R^{size}
  // norm \in R
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex == 0) {
    for (int i = 0; i < size; ++ i) {
      norm[0] += pow(A[i], 2);
    }
    norm[0] = sqrt(norm[0]);
  }
}

KERNEL
void elementwise_prod(const double * A, const double * B, double * D, const int size) {
  // D = A . B
  // D, A, B \in R^{size}
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex < size) {
    D[linearIndex] = A[linearIndex] * B[linearIndex];
  }
}

KERNEL
void elementwise_subtract(const double * A, const double * B, double * D, const int size) {
  // D = A + B
  // D, A, B \in R^{size}
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex < size) {
    D[linearIndex] = A[linearIndex] - B[linearIndex];
  }
}

KERNEL
void elementwise_add(const double * A, const double * B, double * D, const int size) {
  // D = A + B
  // D, A, B \in R^{size}
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex < size) {
    D[linearIndex] = A[linearIndex] + B[linearIndex];
  }
}

KERNEL
void scalar_prod(const double * A, const double * s, double * D, const int size) {
  // D = A . s
  // D, A \in R^{size}
  // s \in R
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex < size) {
    D[linearIndex] = A[linearIndex] * s[0];
  }
}

KERNEL
void scalar_prod(const double * A, const double s, double * D, const int size) {
  // D = A . s
  // D, A \in R^{size}
  // s \in R
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex < size) {
    D[linearIndex] = A[linearIndex] * s;
  }
}

KERNEL
void scalar_div(const double * A, const double * s, double * D, const int size) {
  // D = A / s
  // D, A \in R^{size}
  // s \in R
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex < size && s != 0) {
    D[linearIndex] = A[linearIndex] / s[0];
  }
}

KERNEL
void scalar_div(const double * A, const double s, double * D, const int size) {
  // D = A / s
  // D, A \in R^{size}
  // s \in R
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex < size && s != 0) {
    D[linearIndex] = A[linearIndex] / s;
  }
}

KERNEL
void matmul(const double * A, const double * B, double * C, const int m, const int n, const int k) {
  // C = A B
  // A \in R^{m x k}
  // B \in R^{k x n}
  // C \in R^{m x n}
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearIndex < m * n) {
    const int y = linearIndex % n;
    const int x = linearIndex / n;

    C[linearIndex] = 0.0;

    for (int z=0; z < k; ++z) {
      C[linearIndex] += A[x * k + z] * B[z * n + y]; 
    }
  }
}

template <typename T = double>
KERNEL
void fill(T * data, const T val, const int size) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex < size) {
    data[linearIndex] = val;
  }
}

template <typename T = double>
HOST
T* zeros_cpu(const int m, const int k) {
  int size = m * k;
  T * mat = (T *)malloc(sizeof(T) * size);

  // Fill 
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      mat[i * k + j] = 0.0;
    }
  }

  return mat;
}

template <typename T = double>
HOST
T* zeros_cuda(const int size) {
  T *mat;
  cudaMalloc(&mat, sizeof(T) * size);

  // Fill kernel call
  int max_threads = 128;
  int blocks = (size + max_threads - 1) / max_threads;
  fill<T><<<blocks, max_threads>>>(mat, 0.0, size);
  cudaDeviceSynchronize(); // We call this on host (CPU) to wait for threads to finish their work.

  return mat;
}

template <typename T = double>
HOST
T* zeros_cuda(const int m, const int k) {
  int size = m * k;
  T *mat;
  cudaMalloc(&mat, sizeof(T) * size);

  // Fill kernel call
  int max_threads = 4;
  int blocks = (size + max_threads - 1) / max_threads;
  fill<<<blocks, max_threads>>>(mat, 0.0, size);
  cudaDeviceSynchronize(); // We call this on host (CPU) to wait for threads to finish their work.

  return mat;
}

HOST
double* dot_product(const double* a, const double* b, const int size) {
  double* d = zeros_cuda(1, size);
  double* dot_prod = zeros_cuda(1, 1);

  // Hadamard product kernel call
  int max_threads = 4;
  int blocks = (size + max_threads - 1) / max_threads;
  elementwise_prod<<<blocks, max_threads>>>(a, b, d, size);
  cudaDeviceSynchronize(); // We call this on host (CPU) to wait for threads to finish their work.

  // Sum kernel call
  sum<<<1, 1>>>(d, dot_prod, size);
  cudaDeviceSynchronize(); // We call this on host (CPU) to wait for threads to finish their work.
  return dot_prod;
}

HOST
double* normalize(const double* vec, const int size) {
  double* norm = zeros_cuda(1, 1);
  double* vec_out = zeros_cuda(1, size);

  // Calculate norm with no parallelism
  // We only do it this way because it's not worth the parallelism in our typical use case (size <= 4)
  l2_norm<<<1, 1>>>(/* vector = */ vec, norm, /* size = */ size);
  cudaDeviceSynchronize(); // We call this on host (CPU) to wait for threads to finish their work.

  // Normalize kernel call
  int max_threads = 4;
  int blocks = (size + max_threads - 1) / max_threads;
  scalar_div<<<blocks, max_threads>>>(vec, norm, vec_out, size);
  cudaDeviceSynchronize(); // We call this on host (CPU) to wait for threads to finish their work.
  return vec_out;
}

KERNEL
void device_transform_kernel(double * mat, const double height, const double width) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearIndex == 0) {
    double scale_h = height / 2;
    double scale_w = width / 2;
    mat[INDEX4(0, 0)] = scale_w;
    mat[INDEX4(1, 1)] = scale_h;
    mat[INDEX4(2, 2)] = 1.0;
    mat[INDEX4(3, 0)] = scale_w;
    mat[INDEX4(3, 1)] = scale_h;
    mat[INDEX4(3, 3)] = 1.0;
  }
}

KERNEL
void camera_transform_kernel(double * mat, const double * u_, const double * v_, const double * w_,
    const double * ut, const double * vt, const double * wt) {
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
void view_transform_kernel(double * mat, const double angle, const double near, const double far) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearIndex == 0) {
    double cot_alpha_div_2 = cot(angle / 2.0);
    mat[INDEX4(0, 0)] = cot_alpha_div_2;
    mat[INDEX4(1, 1)] = cot_alpha_div_2;
    mat[INDEX4(2, 2)] = (far + near) / (far - near);
    mat[INDEX4(3, 2)] = (2 * far * near) / (far - near);
    mat[INDEX4(2, 3)] = -1.0;
  }
}

double* device_transform(const double height, const double width) {
  double* mat = zeros_cuda(4, 4);
  device_transform_kernel<<<1, 1>>>(mat, height, width);
  cudaDeviceSynchronize();
  return mat;
}

double* camera_transform(Camera camera) {
  double* mat = zeros_cuda(4, 4);
  double* w_ = zeros_cuda(1, 3);
  double* u_ = zeros_cuda(1, 3);
  double* v_ = zeros_cuda(1, 3);
  double* t_ = zeros_cuda(1, 3);

  // Compute w
  elementwise_subtract<<<1, 3>>>(/* A = */ camera.position, /* B = */ camera.focus, /* out = */ w_, /* size = */ 3);
  cudaDeviceSynchronize(); // We call this on host (CPU) to wait for threads to finish their work.
  w_ = normalize(/* vec = */ w_, /* size = */ 3);

  // Compute u
  cross3<<<1, 1>>>(camera.up, w_, /* out = */ u_);
  cudaDeviceSynchronize(); // We call this on host (CPU) to wait for threads to finish their work.
  u_ = normalize(/* vec = */ u_, /* size = */ 3);

  // Compute v
  cross3<<<1, 1>>>(w_, u_, /* out = */ v_);
  cudaDeviceSynchronize(); // We call this on host (CPU) to wait for threads to finish their work.
  // No need to normalize v because it's the cross product of two unit vectors

  // Compute t
  scalar_prod<<<1, 3>>>(camera.position, -1.0, t_, 3);
  cudaDeviceSynchronize(); // We call this on host (CPU) to wait for threads to finish their work.

  // Compute u t
  double* ut = dot_product(u_, t_, 3);
  double* vt = dot_product(v_, t_, 3);
  double* wt = dot_product(w_, t_, 3);

  // Arrange values
  camera_transform_kernel<<<1, 1>>>(mat, u_, v_, w_, ut, vt, wt);
  cudaDeviceSynchronize(); // We call this on host (CPU) to wait for threads to finish their work.

  return mat;
}

double* view_transform(Camera camera) {
  double* mat = zeros_cuda(4, 4);
  view_transform_kernel<<<1, 1>>>(mat, camera.angle, camera.near, camera.far);
  cudaDeviceSynchronize(); // We call this on host (CPU) to wait for threads to finish their work.
  return mat;
}

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

template <typename T = double>
HOST
T* to_cpu(T * data_cuda, const int size, const bool free_src = false) {
  T * data_cpu = (T *)malloc(sizeof(T) * size);
  cudaMemcpy(data_cpu, data_cuda, size * sizeof(T), cudaMemcpyDeviceToHost);
  if (free_src)
    cudaFree(data_cuda);
  return data_cpu;
}

HOST
double* to_cuda(double * data_cpu, const int size, const bool free_src = false) {
  double *data_cuda;
  cudaMalloc(&data_cuda, sizeof(double) * size);
  cudaMemcpy(data_cuda, data_cpu, size * sizeof(double), cudaMemcpyHostToDevice);
  if (free_src)
    free(data_cpu);
  return data_cuda;
}

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
  elementwise_subtract<<<1, 3>>>(/* A = */ c.position, /* B = */ c.focus, /* out = */ lightDir, /* size = */ 3);
  cudaDeviceSynchronize(); // We call this on host (CPU) to wait for threads to finish their work.
  lightDir = normalize(/* vec = */ lightDir, /* size = */ 3);

  return LightingParameters(lightDir, Ka, Kd, Ks, alpha);
}

HOST
double* GetTransforms(Camera camera, const double height, const double width) {
  double* D = device_transform(height, width);
  double* C = camera_transform(camera);
  double* V = view_transform(camera);
  double* CV = zeros_cuda(4, 4);
  double* output = zeros_cuda(4, 4);
  // Perform 2 matrix multiplications
  // Both 4x4 inputs and 4x4 output
  // So we'd need 16 threads, one for each output scalar.
  matmul<<<1, 16>>>(C, V, CV, 4, 4, 4);
  matmul<<<1, 16>>>(CV, D, output, 4, 4, 4);
  cudaDeviceSynchronize();
  return output;
}

DEVICE
double * device_view_direction(const double * c, const double * v) {
  // We're using malloc and not cudaMalloc in device code.
  double * out = (double *)(malloc(sizeof(double) * 3));
  double norm = 0;
  for (int i=0; i < 3; ++i) {
    double acc = c[i] - v[i];
    out[i] = acc;
    norm += pow(acc, 2);
  }
  norm = sqrt(norm);
  for (int i=0; i < 3; ++i) {
    out[i] = out[i] / norm;
  }
  return out;
}

DEVICE
double * device_elementwise_prod(const double * a, const double b, const int n) {
  // We're using malloc and not cudaMalloc in device code.
  double * out = (double *)(malloc(sizeof(double) * n));

  for (int i=0; i < n; ++i) {
    out[i] = a[i] * b;
  }
  return out;
}

DEVICE
double * device_elementwise_subtract(const double * a, const double * b, const int n) {
  // We're using malloc and not cudaMalloc in device code.
  double * out = (double *)(malloc(sizeof(double) * n));

  for (int i=0; i < n; ++i) {
    out[i] = a[i] - b[i];
  }
  return out;
}

DEVICE
double * device_elementwise_prod_and_subtract(const double * a, const double * b, const double s, const int n) {
  // We're using malloc and not cudaMalloc in device code.
  double * out = (double *)(malloc(sizeof(double) * n));

  for (int i=0; i < n; ++i) {
    out[i] = s * a[i] - b[i];
  }
  return out;
}

KERNEL
void phong_shader(
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
    double LN = device_dot_product(light_direction, vertex_normals + linearIndex * 3, 3);
    double diffuse = max(0.0, LN);

    double * view_direction = device_view_direction(camera_position, vertex_positions + linearIndex * 3);

    double * A = device_elementwise_prod(vertex_normals + linearIndex * 3, (2*LN), 3);
    double * R = device_elementwise_subtract(A, light_direction, 3);
    //double * R = device_elementwise_prod_and_subtract(vertex_normals + linearIndex * 3, light_direction, 2*LN, 3);
    double RV = max(0.0, device_dot_product(R, view_direction, 3));
    free(A);
    free(R);
    free(view_direction);

    double specular = pow(RV, alpha);
    shading_values[linearIndex] = Ka + Kd * diffuse + Ks * specular;
  }
}

KERNEL
void mvp_transform(
    const double * vertex_positions,
    const double * mvp,
    double * out_vertex_positions,
    const int num_vertices) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = linearIndex / 3;
  const int j = linearIndex % 3;
  if (i < num_vertices) {
    double w = mvp[3 * 4 + 3]; // 4th dimension
    double accum = mvp[3 * 4 + j]; // 1.0 column
    for (int k=0; k < 3; ++k) {
      accum += vertex_positions[i * 3 + k] * mvp[k * 4 + j];
      w += vertex_positions[i * 3 + k] * mvp[k * 4 + 3];
    }
    out_vertex_positions[i * 3 + j] = accum / w;
  }
}

HOST
void shader_and_transform(Model *m, Camera c, LightingParameters lp, double height, double width) {
  double* mvp = GetTransforms(c, height, width);
  int num_vertices = m->numTriangles * 3;
  int problem_size = m->numTriangles * 9;

  cudaMalloc(&m->shading, sizeof(double) * num_vertices);

  #define SHADER_NUM_THREADS 512
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

  cudaMalloc(&m->out_vertices, sizeof(double) * problem_size);

  #define MVP_NUM_THREADS 512
  blocks = (problem_size + MVP_NUM_THREADS - 1) / MVP_NUM_THREADS;
  threads = MVP_NUM_THREADS;
  mvp_transform<<<blocks, threads>>>(m->vertices, mvp, m->out_vertices, num_vertices);
  cudaDeviceSynchronize();
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
    int max_threads = 128;
    int blocks = (hw3 + max_threads - 1) / max_threads;
    fill<unsigned char><<<blocks, max_threads>>>(data, 0, hw3);
    cudaDeviceSynchronize(); // We call this on host (CPU) to wait for threads to finish their work.
  }

  HOST
  void clear_zbuffer() {
    // Fill kernel call
    int max_threads = 128;
    int blocks = (hw + max_threads - 1) / max_threads;
    fill<<<blocks, max_threads>>>(z_buffer, 0.0, hw);
    cudaDeviceSynchronize(); // We call this on host (CPU) to wait for threads to finish their work.
  }

  //DEVICE HOST
  //void set_pixel(int i, int j, double z, double r, double g, double b) {
  //  if (i < 0 || j < 0 || i >= height || j >= width)
  //    return;
  //  const int pixelIndex = i * width + j;
  //  #ifdef  __CUDA_ARCH__
  //  gpuAtomicMin(z_buffer + pixelIndex, z);
  //  #else
  //  z_buffer[pixelIndex] = min(z_buffer[pixelIndex], z);
  //  #endif
  //  if (z_buffer[pixelIndex] == z) {
  //    data[pixelIndex * 3 + 0] = r;
  //    data[pixelIndex * 3 + 1] = g;
  //    data[pixelIndex * 3 + 2] = b;
  //  }
  //}
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
  double * z_buffer = zeros_cpu(height, width);

  return Image(image_data, z_buffer, height, width);
}

HOST 
Image image_cuda(int height, int width) {
  unsigned char * image_data = zeros_cuda<unsigned char>(height * width * 3);
  double * z_buffer = zeros_cuda(height * width);

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
  double m, b, x_left, x_right;

  DEVICE
  Line(): m(0), b(0), x_left(-1), x_right(-1) {}
  DEVICE
  Line (double m, double b, double x_left, double x_right): m(m), b(b), x_left(x_left), x_right(x_right) {}

  DEVICE
  double intersect(double y) {
    if (m == 0){
      assert(x_left >= 0 && x_right >= 0 && x_left == x_right);
      return x_left;
    }
    return (y - b) / m;
  }

  DEVICE
  double leftIntersection(double y) {
    return (m == 0) ? (x_left) : ((y - b) / m);
  }

  DEVICE
  double rightIntersection(double y) {
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
Line intercept(const double * a, const double * b) {
  if (abs_difference(a[0], b[0]) == 0) {
      // Horizontal line -- to prevent zero division, we just return the leftmost and rightmost X coordinates.
      return Line(0, 0, min(a[0], b[0]), max(a[0], b[0]));
  }
  if (abs_difference(a[1], b[1]) == 0) {
      return Line(); // Vertical lines are considered invalid.
  }
  double m_ = (b[1] - a[1]) / (b[0] - a[0]);
  double b_ = b[1] - (m_ * b[0]);
  return Line(m_, b_, -1, -1);
}

DEVICE
void set_pixel_cuda(unsigned char * data, double * z_buffer, const int height, const int width, 
    int i, int j, double z, double r, double g, double b) {
  if (i < 0 || j < 0 || i >= height || j >= width)
    return;
  const int pixelIndex = i * width + j;
  #ifdef  __CUDA_ARCH__
  gpuAtomicMin(z_buffer + pixelIndex, z);
  #else
  z_buffer[pixelIndex] = min(z_buffer[pixelIndex], z);
  #endif
  if (z_buffer[pixelIndex] == z) {
  printf("%d %d  -- color: %f %f %f \n", i, j, r, g, b);
    data[pixelIndex * 3 + 0] = r;
    data[pixelIndex * 3 + 1] = g;
    data[pixelIndex * 3 + 2] = b;
  }
}

DEVICE
double * color_lerp(const double coord_A, const double coord_B, const double coord_C,
    const double * color_A, const double * color_B) {
  // We're using malloc and not cudaMalloc in device code.
  double * out = (double *)(malloc(sizeof(double) * 3));
  double t = (coord_C - coord_A) / (coord_B - coord_A);
  for (int i = 0; i < 3; ++i) {
    out[i] = color_A[i] * t + color_B[i] * (1 - t);
  }
  return out;
}

DEVICE
double * color_lerp(const double * coord_A, const double * coord_B, const double * coord_C,
    const double * color_A, const double * color_B) {
  // We're using malloc and not cudaMalloc in device code.
  double * out = (double *)(malloc(sizeof(double) * 3));

  double f_ba = 0.0;
  double f_ca = 0.0;
  for (int i=0; i < 2; ++i) {
    f_ba += pow(coord_B[i] - coord_A[i], 2);
    f_ca += pow(coord_C[i] - coord_A[i], 2);
  }
  f_ba = sqrt(f_ba);
  f_ca = sqrt(f_ca);
  double t = f_ca / f_ba;
    
  for (int i = 0; i < 3; ++i) {
    out[i] = color_A[i] * t + color_B[i] * (1 - t);
  }

  return out;
}

DEVICE
double scalar_lerp(const double * coord_A, const double * coord_B, const double * coord_C,
    const double val_A, const double val_B) {
  double f_ba = 0.0;
  double f_ca = 0.0;
  for (int i=0; i < 2; ++i) {
    f_ba += pow(coord_B[i] - coord_A[i], 2);
    f_ca += pow(coord_C[i] - coord_A[i], 2);
  }
  f_ba = sqrt(f_ba);
  f_ca = sqrt(f_ca);
  double t = f_ca / f_ba;
  return val_A * t + val_B * (1 - t);
}

DEVICE
double scalar_lerp(const double coord_A, const double coord_B, const double coord_C,
    const double val_A, const double val_B) {
  double t = (coord_C - coord_A) / (coord_B - coord_A);
  return val_A * t + val_B * (1 - t);
}

DEVICE
double * make_coord2d(const double A, const double B) {
  double * out = (double *)(malloc(sizeof(double) * 2));
  out[0] = A;
  out[1] = B;
  return out;
}

DEVICE
void scanline(
    const double * position,
    const double * color,
    const double * shading,
    unsigned char * image_data,
    double * z_buffer,
    const int height,
    const int width,
    int anchorIdx,
    int leftIdx,
    int rightIdx,
    int minIdx,
    int maxIdx) {
  const double * anchor = position + anchorIdx * 3;
  const double * left   = position + leftIdx   * 3;
  const double * right  = position + rightIdx  * 3;
  double * anchorColor = (double*)(color + anchorIdx * 3);
  double *   leftColor = (double*)(color + leftIdx   * 3);
  double *  rightColor = (double*)(color + rightIdx  * 3);
  double anchorShading = shading[anchorIdx];
  double   leftShading = shading[leftIdx  ];
  double  rightShading = shading[rightIdx ];
  double rowMin  = position[minIdx * 3 + /* y offset is 1 */ 1];
  double rowMax  = position[maxIdx * 3 + /* y offset is 1 */ 1];
  //printf("%f %f \n", rowMin, rowMax);

  /* Scanline */
  Line leftEdge  = intercept( left, anchor);
  Line rightEdge = intercept(right, anchor);
  if (leftEdge.valid() && rightEdge.valid()) {
    for (int r=C441(rowMin); r <= F441(rowMax); ++r) {
      double leftEnd  =   leftEdge.leftIntersection(r);
      double rightEnd = rightEdge.rightIntersection(r);
      double * leftC = make_coord2d(leftEnd, r);
      double * rightC = make_coord2d(rightEnd, r);
      double leftZ = scalar_lerp(
          left, 
          anchor, 
          leftC,
          left[2], 
          anchor[2]);
      double rightZ = scalar_lerp(
          right, 
          anchor, 
          rightC,
          right[2], 
          anchor[2]);
      double * leftColorX = color_lerp(
          left, 
          anchor, 
          leftC,
          leftColor,
          anchorColor);
      double * rightColorX = color_lerp(
          right, 
          anchor, 
          rightC,
          rightColor,
          anchorColor);
      double leftShadingX = scalar_lerp(
          left, 
          anchor, 
          leftC,
          leftShading, 
          anchorShading);
      double rightShadingX = scalar_lerp(
          right, 
          anchor, 
          rightC,
          rightShading, 
          anchorShading);
      free(leftC);
      free(rightC);
      //if (leftEnd >= rightEnd) {
      //  swap<double>(&leftZ, &rightZ);
      //  swap<Color>(&leftColorX, &rightColorX);
      //  swap<double>(&leftShadingX, &rightShadingX);
      //  swap<double>(&leftEnd, &rightEnd);
      //}
      //printf("%f %f   --- %f %f \n", rowMin, rowMax, leftEnd, rightEnd);
      for (int c = C441(leftEnd); c <= F441(rightEnd); ++c) {
        double z = scalar_lerp(
             leftEnd, 
            rightEnd, 
            c,
            leftZ, 
            rightZ);
        double * color = color_lerp(
            leftEnd, 
            rightEnd, 
            c,
            leftColorX,
            rightColorX);
        double shading = scalar_lerp(
             leftEnd, 
            rightEnd, 
            c,
            leftShadingX, 
            rightShadingX);
        set_pixel_cuda(image_data, z_buffer, height, width, c, r, z, 
              C441(255.0 * min(max(0.0, color[0] * shading), 1.0)), 
              C441(255.0 * min(max(0.0, color[1] * shading), 1.0)), 
              C441(255.0 * min(max(0.0, color[2] * shading), 1.0))
              );
        free(color);
      }
      free(leftColorX);
      free(rightColorX);
    }
  }
}

KERNEL
void rasterization_kernel(
    const double * vertex_positions, 
    const double * vertex_colors,
    const double * vertex_shadings,
    //const int * triangle_to_vertex,
    unsigned char * image_data,
    double * z_buffer,
    const int height,
    const int width,
    const int num_triangles
    ) {
  const int triangleIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (triangleIdx < num_triangles) {
    const int vertexOffset = triangleIdx * 9;
    const int shadingOffset = triangleIdx * 3;
    const double * position = vertex_positions + vertexOffset;
    const double * colors   = vertex_colors    + vertexOffset;
    const double * shadings = vertex_shadings  + shadingOffset;

    //const int * vertIdx = triangle_to_vertex + triangleIdx * 7;
    //const int topv = vertIdx[0];
    //const int midv = vertIdx[1];
    //const int botv = vertIdx[2];
    //const int  tlv = vertIdx[3];
    //const int  trv = vertIdx[4];
    //const int  blv = vertIdx[5];
    //const int  brv = vertIdx[6];
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

    //printf("T: %d   %f %f %f\n", triangleIdx, position[topv*3+1], position[midv*3+1], position[botv*3+1]);
    scanline(position, colors, shadings, image_data, z_buffer, height, width, botv, blv, brv, /* rowMin = */botv, /* rowMax = */midv);
    scanline(position, colors, shadings, image_data, z_buffer, height, width, topv, tlv, trv, /* rowMin = */midv, /* rowMax = */topv);
  }
}

HOST
void rasterize(Image * image, Model model) {
  // To each triangle its own thread
  int problem_size = model.numTriangles;

  #define RASTERIZER_NUM_THREADS 128
  int blocks = (problem_size + RASTERIZER_NUM_THREADS - 1) / RASTERIZER_NUM_THREADS;
  int threads = RASTERIZER_NUM_THREADS;
  rasterization_kernel<<<blocks, threads>>>(
      model.out_vertices,
      model.colors,
      model.shading,
      //model.t_to_v,
      image->data,
      image->z_buffer,
      image->height,
      image->width,
      problem_size);
  cudaDeviceSynchronize();
  //double *a = zeros_cuda(1,1);
  //blocks = (image->height*image->width*3 + threads - 1) / threads;
  //sum<<<blocks, threads>>>(image->data, a, image->hw3);
  //cudaDeviceSynchronize();
  //double * a_cpu = to_cpu(a, 1);
  //std::cout << "SUM: " << a_cpu[0] << "\n";
}

HOST 
void PrintMat(double * mat, const int m, const int k) {
  const int size = m * k;
  double * mat_cpu = to_cpu(mat, size);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      std::cout << " " << mat_cpu[i * k + j] << ",";
    }
    std::cout << std::endl;
  }
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
  double height = 1000;
  double width = 1000;
  double f = 0;
  Image image = image_cuda(height, width);
  Model model = ReadTriangles();
  Camera camera = GetCamera(f, 1000);
  LightingParameters lp = GetLighting(camera);
  shader_and_transform(&model, camera, lp, height, width);
  CHECK_LAST_CUDA_ERROR();
  image.clear_zbuffer();
  CHECK_LAST_CUDA_ERROR();
  rasterize(&image, model);
  CHECK_LAST_CUDA_ERROR();
  if (image.on_device) {
    Image2PNM(image_to_cpu(image), gen_filename(f));
  } else {
    Image2PNM(image, gen_filename(f));
  }
  CHECK_LAST_CUDA_ERROR();
  return 0;
}
