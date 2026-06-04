NVCC ?= /usr/local/cuda/bin/nvcc
CUDA_ARCH ?= sm_70
BUILD_DIR ?= build/

CPPFLAGS ?= -Iinclude
NVCCFLAGS ?= -std=c++17 -O2 -arch=$(CUDA_ARCH)

SOURCES := $(shell find src/algos -type f -name '*.cu' | sort)
KERNELS := $(patsubst src/algos/%.cu,$(BUILD_DIR)/%,$(SOURCES))

.PHONY: all kernels help manifest clean list

all: kernels

kernels: $(KERNELS)

$(BUILD_DIR)/%: src/algos/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $< -o $@

help:
	@echo "Available targets:"
	@echo "  all, kernels  Build all CUDA mining kernel executables"
	@echo "  list          Print discovered CUDA kernel sources"
	@echo "  manifest      Regenerate source manifest"
	@echo "  clean         Remove build outputs and temporary files"
	@echo
	@echo "Variables:"
	@echo "  NVCC=$(NVCC)"
	@echo "  CUDA_ARCH=$(CUDA_ARCH)"
	@echo "  BUILD_DIR=$(BUILD_DIR)"

list:
	@printf '%s\n' $(SOURCES)

clean:
	rm -rf build

