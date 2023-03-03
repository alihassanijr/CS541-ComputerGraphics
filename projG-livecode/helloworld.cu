/*

CS 441/541

Hello World!

Reach out to alih@uoregon.edu if you have any questions.

*/
#include <cuda.h>

#include <iostream>
#include <string>
#include <assert.h>

// Just a function that kills our program if there's a cuda failure.
void showCudaError(){
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error: " << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

///////////////////////////////////////////////////////////////////////////
////////////////////////////// CPU Functions //////////////////////////////
///////////////////////////////////////////////////////////////////////////

// Matrix multiplication
void matrix_multiply_cpu(
    const double * matrix_A, 
    const double * matrix_B,
    double * matrix_C,
    const int M,
    const int N,
    const int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      const int linearIndex = i * N + j;
      matrix_C[linearIndex] = 0;
      for (int k=0; k < K; ++k) {
        matrix_C[linearIndex] += matrix_A[i * K + k] * matrix_B[k * M + j];
      }
    }
  }
}

// Array/Matrix filler
void fill_cpu(double * mat, int size) {
  for (int i=0; i < size; ++i)
    mat[i] = double(1.0 / double(i+1));
}

// Matrix printer
void print_cpu(double * mat, int M, int N, std::string s) {
  printf("%s = [ \n", s.c_str());
  for (int i=0; i < M; ++i) {
    for (int j=0; j < N; ++j) {
      printf("  %f", mat[i * N + j]);
    }
    printf("\n");
  }
  printf("] \n");
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
////////////////////////////// GPU Functions //////////////////////////////
///////////////////////////////////////////////////////////////////////////

// Matrix multiplication filler kernel (GPU)
__global__
void matrix_multiply_kernel(
    const double * matrix_A, 
    const double * matrix_B,
    double * matrix_C,
    const int M,
    const int N,
    const int K) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex < M * N) {
    // Figure out i and j
    const int i = linearIndex / N;
    const int j = linearIndex % N;

    matrix_C[linearIndex] = 0;
    for (int k=0; k < K; ++k) {
      matrix_C[linearIndex] += matrix_A[i * K + k] * matrix_B[k * M + j];
    }
  }
}

// Array/Matrix filler kernel (GPU)
__global__
void fill_cuda(double * mat, int size) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearIndex < size) {
    mat[linearIndex] = double(1.0 / double(linearIndex+1));
  }
}

// Device (GPU) matrix printer
// Transfers matrix from device to host, then calls CPU printer.
void print_cuda(double * mat, int M, int N, std::string s) {
  double * mat_cpu = (double *) malloc(M * N * sizeof(double));
  cudaMemcpy(mat_cpu, mat, M * N * sizeof(double), cudaMemcpyDeviceToHost);
  print_cpu(mat_cpu, M, N, s);
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

// Our main function
int main() {
  int M = 5000;
  int N = 6000;
  int K = 7000;

  double * matrix_A;
  double * matrix_B;
  double * matrix_C;
  #ifdef USE_CUDA
    int size = M * N;
    int THREADS = 256;
    //int BLOCKS = (size + THREADS - 1) / THREADS;
    cudaMalloc(&matrix_A, M * K * sizeof(double));
    cudaMalloc(&matrix_B, K * N * sizeof(double));
    cudaMalloc(&matrix_C, M * N * sizeof(double));
    int BLOCKS = (M * K + THREADS - 1) / THREADS;
    fill_cuda<<<BLOCKS, THREADS>>>(matrix_A, M * K);
    BLOCKS = (K * N + THREADS - 1) / THREADS;
    fill_cuda<<<BLOCKS, THREADS>>>(matrix_B, K * N);
    cudaDeviceSynchronize();
    showCudaError();
    BLOCKS = (size + THREADS - 1) / THREADS;
    matrix_multiply_kernel<<<BLOCKS, THREADS>>>(matrix_A, matrix_B, matrix_C, M, N, K);
    cudaDeviceSynchronize();
    showCudaError();
    //print_cuda(matrix_A, M, K, "A");
    //print_cuda(matrix_B, K, N, "B");
    //print_cuda(matrix_C, M, N, "C");
  #else
    matrix_A = (double*) malloc(M * K * sizeof(double));
    matrix_B = (double*) malloc(K * N * sizeof(double));
    matrix_C = (double*) malloc(M * N * sizeof(double));
    fill_cpu(matrix_A, M * K);
    fill_cpu(matrix_B, K * N);
    matrix_multiply_cpu(matrix_A, matrix_B, matrix_C, M, N, K);
    //print_cpu(matrix_A, M, K, "A");
    //print_cpu(matrix_B, K, N, "B");
    //print_cpu(matrix_C, M, N, "C");
  #endif
  return 0;
}
