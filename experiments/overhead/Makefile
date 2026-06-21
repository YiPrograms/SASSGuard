CXX = g++
CUDA_PATH ?= /usr/local/cuda
PTXAS ?= $(CUDA_PATH)/bin/ptxas
DETECTED_CUDA_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.' | sed 's/^/sm_/')
CUDA_ARCH ?= $(if $(DETECTED_CUDA_ARCH),$(DETECTED_CUDA_ARCH),sm_70)
RUNS ?= 1000
PROGRESS_EVERY ?= 10
RUN_TIMEOUT_SECONDS ?= 120
MODULE := matSumKernel.cubin
PTX := matSumKernel.ptx

drivertest: drivertest.cpp $(MODULE)
	$(CXX) drivertest.cpp -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -lcuda -o drivertest

$(MODULE): $(PTX)
	$(PTXAS) -arch=$(CUDA_ARCH) $< -o $@

.PHONY: cubin median overhead-median clean
cubin: $(MODULE)

median overhead-median: drivertest
	PROGRESS_EVERY=$(PROGRESS_EVERY) RUN_TIMEOUT_SECONDS=$(RUN_TIMEOUT_SECONDS) ./median_overhead.sh $(RUNS)

clean:
	rm -f drivertest $(MODULE)
