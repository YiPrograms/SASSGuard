#include <cuda.h>
#include <cudaTypedefs.h>
#include <cstring>

#include "hook.h"
#include "debug.h"

static CUdevice get_context_device() {
  CUdevice device;
	PFN_cuCtxGetDevice_v2000 ctxGetDevice = (PFN_cuCtxGetDevice_v2000)real_cuCtxGetDevice;
	ctxGetDevice(&device);
  return device;
}

static void get_device_pci_bus_id(char *pci_bus_id, size_t pci_bus_id_size) {
  CUdevice device = get_context_device();

	PFN_cuDeviceGetPCIBusId_v4010 deviceGetPCIBusId = (PFN_cuDeviceGetPCIBusId_v4010)real_cuDeviceGetPCIBusId;
  deviceGetPCIBusId(pci_bus_id, pci_bus_id_size, device);
}

#ifdef __cplusplus
extern "C" {
#endif

#undef cuGetProcAddress
CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {
	PFN_cuGetProcAddress_v11030 real = (PFN_cuGetProcAddress_v11030)real_cuGetProcAddress;

	DEBUG("cuGetProcAddress() searching for symbol %s", symbol);

  // Look for our hooked function
  void *func = get_hooked_function(symbol);

  // Not being hooked
  if (func == NULL) {
    return real(symbol, pfn, cudaVersion, flags);
  }

	DEBUG("cuGetProcAddress() hooked symbol %s", symbol);

	*pfn = func;
	return CUDA_SUCCESS;
}

#undef cuGetProcAddress_v2
CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
														 cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus) {
	PFN_cuGetProcAddress_v12000 real = (PFN_cuGetProcAddress_v12000)real_cuGetProcAddress_v2;

	DEBUG("cuGetProcAddress_v2() searching for symbol %s", symbol);

  // Look for our hooked function
  void *func = get_hooked_function(symbol);

  // Not being hooked
  if (func == NULL) {
    return real(symbol, pfn, cudaVersion, flags, symbolStatus);
  }

	DEBUG("cuGetProcAddress_v2() hooked symbol %s", symbol);

	*pfn = func;
  if (symbolStatus != NULL)
    *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;

	return CUDA_SUCCESS;
}

#undef cuLaunchKernel
CUresult cuLaunchKernel(CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
	PFN_cuLaunchKernel_v4000 real = (PFN_cuLaunchKernel_v4000)real_cuLaunchKernel;

  char pci_bus_id[13];
  get_device_pci_bus_id(pci_bus_id, sizeof(pci_bus_id));

  DEBUG("cuLaunchKernel(): Function %p, gridDim (%u, %u, %u), blockDim (%u, %u, %u), sharedMemBytes %u, stream %p, device %s",
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
        sharedMemBytes, hStream, pci_bus_id);

	return real(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
		sharedMemBytes, hStream, kernelParams, extra);
}

#ifdef __cplusplus
}
#endif

static void *get_hooked_function(const char *symbol) {
  #define HOOK_FUNCTION(fn) \
    if (strcmp(symbol, #fn) == 0) return (void *)(&fn);

  HOOK_FUNCTION(cuLaunchKernel)
  // HOOK_FUNCTION(cuLaunchKernelEx)
  // HOOK_FUNCTION(cuLaunchCooperativeKernel)
  HOOK_FUNCTION(cuGetProcAddress)
  HOOK_FUNCTION(cuGetProcAddress_v2)
  // HOOK_FUNCTION(cuModuleLoad)
  // HOOK_FUNCTION(cuModuleLoadData)
  // HOOK_FUNCTION(cuModuleLoadDataEx)
  // HOOK_FUNCTION(cuModuleLoadFatBinary)
  // HOOK_FUNCTION(cuLibraryLoadData)
  // HOOK_FUNCTION(cuLibraryLoadFromFile)
  // HOOK_FUNCTION(cuModuleGetFunction)
  // HOOK_FUNCTION(cuLibraryGetKernel)
  // HOOK_FUNCTION(cuLibraryGetModule)

  return NULL;
}
