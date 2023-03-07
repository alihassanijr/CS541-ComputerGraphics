/*

CS 441/541

Redoing 1A in CUDA.

Generate a 300x300 image
divide into a 3x3 grid,

| BLACK | GRAY | WHITE |
|  RED  | GREEN| BLUE  |
| PURPLE| CYAN | YELLOW|

BLACK:    0,   0,   0
GRAY:   128, 128, 128
WHITE:  255, 255, 255

RED:    255,   0,   0
GREEN:    0, 255,   0
BLUE:     0,   0, 255

PURPLE: 255,   0, 255
CYAN:     0, 255, 255
PURPLE: 255, 255,   0

Reach out to alih@uoregon.edu if you have any questions.

*/
#include <cuda.h>

#include <iostream>
#include <string>
#include <assert.h>

/// Image2PNM: Dumps an Image instance into a PNM file.
void Image2PNM(unsigned char * img, int height, int width, std::string fn) {
  FILE *f = fopen(fn.c_str(), "wb");
  assert(f != NULL);
  fprintf(f, "P6\n");
  fprintf(f, "%d %d\n", height, width);
  fprintf(f, "%d\n", 255);
  fwrite(img, height * width, /* size of pixel = */ 3 * sizeof(unsigned char), f);
  fclose(f);
}

__global__ void colorize(unsigned char * img, int height, int width) {
  const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (linearIndex < height * width) {
    const int i = linearIndex / width;
    const int j = linearIndex % width;

    if (i < 100 && j < 100) {
      img[linearIndex * 3 + 0] = 0;
      img[linearIndex * 3 + 1] = 0;
      img[linearIndex * 3 + 2] = 0;
    }
    if (i < 100 && j >= 100 && j < 200) {
      img[linearIndex * 3 + 0] = 128;
      img[linearIndex * 3 + 1] = 128;
      img[linearIndex * 3 + 2] = 128;
    }
    if (i < 100 && j >= 200) {
      img[linearIndex * 3 + 0] = 255;
      img[linearIndex * 3 + 1] = 255;
      img[linearIndex * 3 + 2] = 255;
    }

    if (i >= 100 && i < 200 && j < 100) {
      img[linearIndex * 3 + 0] = 255;
      img[linearIndex * 3 + 1] = 0;
      img[linearIndex * 3 + 2] = 0;
    }
    if (i >= 100 && i < 200 && j >= 100 && j < 200) {
      img[linearIndex * 3 + 0] = 0;
      img[linearIndex * 3 + 1] = 255;
      img[linearIndex * 3 + 2] = 0;
    }
    if (i >= 100 && i < 200 && j >= 200) {
      img[linearIndex * 3 + 0] = 0;
      img[linearIndex * 3 + 1] = 0;
      img[linearIndex * 3 + 2] = 255;
    }

    if (i >= 200 && j < 100) {
      img[linearIndex * 3 + 0] = 255;
      img[linearIndex * 3 + 1] = 0;
      img[linearIndex * 3 + 2] = 255;
    }
    if (i >= 200 && j >= 100 && j < 200) {
      img[linearIndex * 3 + 0] = 0;
      img[linearIndex * 3 + 1] = 255;
      img[linearIndex * 3 + 2] = 255;
    }
    if (i >= 200 && j >= 200) {
      img[linearIndex * 3 + 0] = 255;
      img[linearIndex * 3 + 1] = 255;
      img[linearIndex * 3 + 2] = 0;
    }
  }
}

int main() {
  int height = 300;
  int width = 300;
  int problem_size = height * width;
  int THREADS = 128;
  int BLOCKS = (problem_size + THREADS - 1) / THREADS;
  unsigned char * img_cpu;
  unsigned char * img_gpu;

  img_cpu = (unsigned char *) malloc(sizeof(unsigned char) * problem_size * 3);
  cudaMalloc(&img_gpu, sizeof(unsigned char) * problem_size * 3);

  colorize<<<BLOCKS, THREADS>>>(img_gpu, height, width);

  cudaMemcpy(img_cpu, img_gpu, sizeof(unsigned char) * problem_size * 3, cudaMemcpyDeviceToHost);

  Image2PNM(img_cpu, height, width, "proj1A_cuda.pnm");
  return 0;
}
