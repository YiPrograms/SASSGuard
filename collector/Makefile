CC       := gcc
CXX      := g++
LD       := ld
OBJCOPY  := objcopy
PYTHON   := python3
GO       := go
PROTOC   := protoc
SUDO		 := sudo

DEBUG ?= 1

BUILD ?= build

DRIVER_VERSION := $(shell head -n1 /proc/driver/nvidia/version | grep -Eo "[0-9]+\.[0-9]+\.[0-9]+")
LIBCUDA := /usr/lib/x86_64-linux-gnu/libcuda.so.$(DRIVER_VERSION)
REAL_LIBCUDA := libcuda.so.1.real # Do not clean

TARGET := $(BUILD)/libcuda.so.1
GO_DIR := go
GO_ARCHIVE := $(BUILD)/libsassguard_go.a
GO_SERVER := $(BUILD)/sassguard-collector
GO_ENV := GOCACHE=/tmp/sassguard-go-cache GOMODCACHE=/tmp/sassguard-go-modcache PATH=$(HOME)/go/bin:$(PATH)
CONFIG_STAMP := $(BUILD)/.config-debug-$(DEBUG)

ifeq ($(DEBUG),1)
CFLAGS += -Wall -Wextra -g -fPIC -DCUHOOK_DEBUG
else
CFLAGS += -O2 -Wall -Wextra -g -fPIC
endif

INCLUDES := -Iinclude -Iinclude/vendor
CFLAGS += $(INCLUDES)

CXXFLAGS = $(CFLAGS) -std=c++17

LDFLAGS := -shared -ldl -lpthread

OBJS := $(BUILD)/loader.o $(BUILD)/trampolines.o $(BUILD)/libcuda_blob.o $(BUILD)/hook.o $(BUILD)/code_registry.o $(BUILD)/telemetry.o $(BUILD)/telemetry_bridge.o

all: $(TARGET) $(GO_SERVER) $(TEST)

$(BUILD):
	mkdir -p $(BUILD)

$(CONFIG_STAMP): | $(BUILD)
	rm -f $(BUILD)/.config-debug-*
	touch $@

include/cuda_symbols.def include/cuda_trampolines.inc: $(REAL_LIBCUDA) hooked_symbols scripts/gen_trampoline_symbols.py
	$(PYTHON) scripts/gen_trampoline_symbols.py $(REAL_LIBCUDA)

$(GO_DIR)/protocol/telemetry.pb.go: $(GO_DIR)/proto/telemetry.proto
	cd $(GO_DIR) && $(GO_ENV) $(PROTOC) --go_out=. --go_opt=module=sassguard/hookgo proto/telemetry.proto

# If libcuda.so.1.real doesn't exist, copy the libcuda.so.1 to libcuda.so.1.real.
$(REAL_LIBCUDA):
	$(SUDO) cp $(LIBCUDA) $(REAL_LIBCUDA)

$(BUILD)/libcuda_blob.o: $(REAL_LIBCUDA) | $(BUILD)
	$(LD) -r -b binary $(REAL_LIBCUDA) -o $@

$(BUILD)/loader.o: src/loader.c include/cuda_symbols.def $(CONFIG_STAMP) | $(BUILD)
	$(CC) $(CFLAGS) -c src/loader.c -o $@

$(BUILD)/trampolines.o: src/trampolines.S include/cuda_trampolines.inc $(CONFIG_STAMP) | $(BUILD)
	$(CC) $(CFLAGS) -c src/trampolines.S -o $@

$(BUILD)/hook.o: src/hook.cc include/cuda_symbols.def include/code_registry.h $(CONFIG_STAMP) | $(BUILD)
	$(CXX) $(CFLAGS) -c src/hook.cc -o $@

$(BUILD)/code_registry.o: src/code_registry.cc include/code_registry.h include/telemetry.h $(CONFIG_STAMP) | $(BUILD)
	$(CXX) $(CFLAGS) -c src/code_registry.cc -o $@

$(BUILD)/telemetry.o: src/telemetry.cc include/telemetry.h $(CONFIG_STAMP) | $(BUILD)
	$(CXX) $(CFLAGS) -c src/telemetry.cc -o $@

$(BUILD)/telemetry_bridge.o: src/telemetry_bridge.cc include/telemetry.h include/config.h $(CONFIG_STAMP) | $(BUILD)
	$(CXX) $(CFLAGS) -c src/telemetry_bridge.cc -o $@

$(GO_ARCHIVE): $(GO_DIR)/client/main.go $(GO_DIR)/go.mod $(GO_DIR)/go.sum $(GO_DIR)/protocol/telemetry.pb.go include/telemetry.h | $(BUILD)
	cd $(GO_DIR) && $(GO_ENV) $(GO) build -buildmode=c-archive -o ../$(GO_ARCHIVE) ./client

$(GO_SERVER): $(GO_DIR)/cmd/sassguard-collector/*.go $(GO_DIR)/go.mod $(GO_DIR)/go.sum $(GO_DIR)/protocol/telemetry.pb.go | $(BUILD)
	cd $(GO_DIR) && $(GO_ENV) $(GO) build -o ../$(GO_SERVER) ./cmd/sassguard-collector

$(TARGET): $(OBJS) $(GO_ARCHIVE)
	$(CXX) \
	  -shared \
	  -Wl,-soname,libcuda.so.1 \
	  -Wl,--version-script=exports.map \
		-Wl,-z,noexecstack \
		-static-libgcc -static-libstdc++ \
	  $(OBJS) \
	  $(GO_ARCHIVE) \
	  -ldl -lpthread -lm \
	  -o $(TARGET)

clean:
	rm -rf $(BUILD)
	rm -f include/cuda_symbols.def include/cuda_trampolines.inc

install: $(TARGET)
	$(SUDO) cp $(TARGET) $(LIBCUDA)

restore:
	$(SUDO) cp $(REAL_LIBCUDA) $(LIBCUDA)

.PHONY: all clean install restore
