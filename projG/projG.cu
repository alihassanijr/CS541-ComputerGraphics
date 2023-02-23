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
  double * normals;
  double * colors;
  double * shading;
  double ** t_to_v;
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

  // Triangle to vertex map
  //// Every triangle has 3 pointers to its 3 vertices sorted by y-value,
  //// two pointers to the left and right vertices w.r.t the top vertex,
  //// and two pointers to the left and right vertices w.r.t the bottom vertex.
  //// 3 + 2 + = 7.
  cudaMalloc(&m.t_to_v,   sizeof(double*) * numTriangles * 7);
  
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
    cudaMemcpy(m.vertices + (i * 3),  coords, 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m.colors   + (i * 3),  colors, 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(m.normals  + (i * 3), normals, 9 * sizeof(double), cudaMemcpyHostToDevice);
    double ** host_t2v = (double **)malloc(sizeof(double*) * 7);
    int top_idx = -1;
    int mid_idx = -1;
    int bot_idx = -1;
    int top_left  = -1;
    int top_right = -1;
    int bot_left  = -1;
    int bot_right = -1;
    if (coords[INDEX3(0, 1)] >= max(coords[INDEX3(1, 1)], coords[INDEX3(2, 1)])){
      top_idx = 0;
      if (coords[INDEX3(1, 1)] >= coords[INDEX3(2, 1)]) {
        mid_idx = 1;
        bot_idx = 2;
      }
      else {
        mid_idx = 2;
        bot_idx = 1;
      }
    }
    else if (coords[INDEX3(1, 1)] >= max(coords[INDEX3(0, 1)], coords[INDEX3(2, 1)])){
      top_idx = 1;
      if (coords[INDEX3(0, 1)] >= coords[INDEX3(2, 1)]) {
        mid_idx = 0;
        bot_idx = 2;
      }
      else {
        mid_idx = 2;
        bot_idx = 0;
      }
    }
    else {
      top_idx = 2;
      if (coords[INDEX3(0, 1)] >= coords[INDEX3(1, 1)]) {
        mid_idx = 0;
        bot_idx = 1;
      }
      else {
        mid_idx = 1;
        bot_idx = 0;
      }
    }
    if (coords[INDEX3(mid_idx, 0)] >= coords[INDEX3(bot_idx, 0)]) {
      top_right = mid_idx;
      top_left  = bot_idx;
    } else {
      top_right = bot_idx;
      top_left  = mid_idx;
    }
    if (coords[INDEX3(mid_idx, 0)] >= coords[INDEX3(top_idx, 0)]) {
      bot_right = mid_idx;
      bot_left  = top_idx;
    } else {
      bot_right = top_idx;
      bot_left  = mid_idx;
    }
    host_t2v[0] = m.vertices + (i * 3) + top_idx;
    host_t2v[1] = m.vertices + (i * 3) + mid_idx;
    host_t2v[2] = m.vertices + (i * 3) + bot_idx;
    host_t2v[3] = m.vertices + (i * 3) + top_left;
    host_t2v[4] = m.vertices + (i * 3) + top_right;
    host_t2v[5] = m.vertices + (i * 3) + bot_left;
    host_t2v[6] = m.vertices + (i * 3) + bot_right;
    cudaMemcpy(m.t_to_v  + (i * 3), host_t2v, 7 * sizeof(double*), cudaMemcpyHostToDevice);
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

KERNEL
void fill(double * data, const double val, const int size) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex < size) {
    data[linearIndex] = val;
  }
}

HOST
double* zeros_cpu(const int m, const int k) {
  double size = m * k;
  double * mat = (double *)malloc(sizeof(double) * size);

  // Fill 
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      mat[i * k + j] = 0.0;
    }
  }

  return mat;
}

HOST
double* zeros_cuda(const int m, const int k) {
  double size = m * k;
  double *mat;
  cudaMalloc(&mat, sizeof(double) * size);

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

HOST
double* to_cpu(double * data_cuda, const int size, const bool free_src = false) {
  double * data_cpu = (double *)malloc(sizeof(double) * size);
  cudaMemcpy(data_cpu, data_cuda, size * sizeof(double), cudaMemcpyDeviceToHost);
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
void shader_and_transform(Model m, Camera c, LightingParameters lp, double height, double width) {
  double* mvp = GetTransforms(c, height, width);
  double* out_vertices;
  double* shading_values;
  int num_vertices = m.numTriangles * 3;
  int problem_size = m.numTriangles * 9;

  cudaMalloc(&shading_values, sizeof(double) * num_vertices);

  #define SHADER_NUM_THREADS 512
  int blocks = (problem_size + SHADER_NUM_THREADS - 1) / SHADER_NUM_THREADS;
  int threads = SHADER_NUM_THREADS;
  phong_shader<<<blocks, threads>>>(
      m.vertices, m.normals, 
      lp.lightDir, c.position, 
      shading_values,
      num_vertices,
      lp.Ka,
      lp.Kd,
      lp.Ks,
      lp.alpha);

  cudaMalloc(&out_vertices, sizeof(double) * problem_size);

  #define MVP_NUM_THREADS 512
  blocks = (problem_size + MVP_NUM_THREADS - 1) / MVP_NUM_THREADS;
  threads = MVP_NUM_THREADS;
  mvp_transform<<<blocks, threads>>>(m.vertices, mvp, out_vertices, num_vertices);
  cudaDeviceSynchronize();

  cudaFree(m.vertices);
  m.vertices = out_vertices;
  m.shading = shading_values;
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


int main() {
  double height = 1000;
  double width = 1000;
  Model m = ReadTriangles();
  Camera camera = GetCamera(10, 1000);
  LightingParameters lp = GetLighting(camera);
  shader_and_transform(m, camera, lp, height, width);
  // rasterize
  CHECK_LAST_CUDA_ERROR();
  return 0;
}
