/*
 * drivertest.cpp
 * Vector addition (host code)
 *
 * Andrei de A. Formiga, 2012-06-04
 */

#include <builtin_types.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#define N 102400

#define TIMER(func)                                                        \
  {                                                                        \
    auto start = std::chrono::high_resolution_clock::now();                \
    func;                                                                  \
    auto end = std::chrono::high_resolution_clock::now();                  \
    auto duration =                                                        \
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start); \
    printf("Time (ns) of %s: %ld\n", #func, duration.count());             \
  }

// --- global variables ----------------------------------------------------
CUdevice device;
CUcontext context;
CUmodule module;
CUfunction function;
size_t totalGlobalMem;

char *module_file = (char *)"matSumKernel.cubin";
char *kernel_name = (char *)"matSum";

// --- functions -----------------------------------------------------------
void initCUDA() {
  int deviceCount = 0;
  CUresult err;
  TIMER(cuInit(0));
  int major = 0, minor = 0;

  TIMER(cuDeviceGetCount(&deviceCount));

  // get first CUDA device
  TIMER(cuDeviceGet(&device, 0));
  char name[100];
  TIMER(cuDeviceGetName(name, 100, device));

  TIMER(cuCtxCreate(&context, 0, device));

  TIMER(cuModuleLoad(&module, module_file));

  TIMER(cuModuleGetFunction(&function, module, kernel_name));
}

void setupDeviceMemory(CUdeviceptr *d_a, CUdeviceptr *d_b, CUdeviceptr *d_c) {
  TIMER(cuMemAlloc(d_a, sizeof(int) * N));
  TIMER(cuMemAlloc(d_b, sizeof(int) * N));
  TIMER(cuMemAlloc(d_c, sizeof(int) * N));
}

void releaseDeviceMemory(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c) {
  TIMER(cuMemFree(d_a));
  TIMER(cuMemFree(d_b));
  TIMER(cuMemFree(d_c));
}

void runKernel(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c) {
  void *args[3] = {&d_a, &d_b, &d_c};

  // grid for kernel: <<<N, 1>>>
  TIMER(cuLaunchKernel(function, N, 1, 1,  // Nx1x1 blocks
                       1, 1, 1,            // 1x1x1 threads
                       0, 0, args, 0));
}

int main(int argc, char **argv) {
  int a[N], b[N], c[N];
  CUdeviceptr d_a, d_b, d_c;

  // initialize host arrays
  for (int i = 0; i < N; ++i) {
    a[i] = N - i;
    b[i] = i * i;
  }

  // initialize
  printf("- Initializing...\n");
  initCUDA();

  // allocate memory
  setupDeviceMemory(&d_a, &d_b, &d_c);

  // copy arrays to device
  TIMER(cuMemcpyHtoD(d_a, a, sizeof(int) * N));
  TIMER(cuMemcpyHtoD(d_b, b, sizeof(int) * N));

  // run
  runKernel(d_a, d_b, d_c);

  // copy results to host and report
  TIMER(cuMemcpyDtoH(c, d_c, sizeof(int) * N));
  for (int i = 0; i < N; ++i) {
    if (c[i] != a[i] + b[i])
      printf("* Error at array position %d: Expected %d, Got %d\n", i,
             a[i] + b[i], c[i]);
  }

  // finish
  releaseDeviceMemory(d_a, d_b, d_c);
  return 0;
}
