CC       := gcc
CXX      := g++
LD       := ld
OBJCOPY  := objcopy
PYTHON   := python3
SUDO		 := sudo

DEBUG ?= 1

BUILD ?= build

LIBCUDA := /usr/lib/x86_64-linux-gnu/libcuda.so.1
REAL_LIBCUDA := /usr/lib/x86_64-linux-gnu/libcuda_real.so.1
BLOB_PREFIX := 

TARGET := $(BUILD)/libcuda.so.1

ifeq ($(DEBUG),1)
CFLAGS += -Wall -Wextra -g -fPIC -DCUHOOK_DEBUG
else
CFLAGS += -O2 -Wall -Wextra -g -fPIC
endif

INCLUDES := -Iinclude -Iinclude/vendor
CFLAGS += $(INCLUDES)

CXXFLAGS = $(CFLAGS) -std=c++17

LDFLAGS := -shared -ldl -lpthread

OBJS := $(BUILD)/loader.o $(BUILD)/trampolines.o $(BUILD)/libcuda_blob.o $(BUILD)/hook.o

all: $(TARGET) $(TEST)

$(BUILD):
	mkdir -p $(BUILD)

include/cuda_symbols.def include/cuda_trampolines.inc: $(REAL_LIBCUDA)
	$(PYTHON) scripts/gen_trampoline_symbols.py $(REAL_LIBCUDA)

# If libcuda.so.1.real doesn't exist, copy the libcuda.so.1 to libcuda.so.1.real.
$(REAL_LIBCUDA):
	$(SUDO) cp $(LIBCUDA) $(REAL_LIBCUDA)

$(BUILD)/libcuda_blob.o: $(REAL_LIBCUDA) | $(BUILD)
	$(LD) -r -b binary $(REAL_LIBCUDA) -o $@

$(BUILD)/loader.o: src/loader.c include/cuda_symbols.def | $(BUILD)
	$(CC) $(CFLAGS) -c src/loader.c -o $@

$(BUILD)/trampolines.o: src/trampolines.S include/cuda_trampolines.inc | $(BUILD)
	$(CC) $(CFLAGS) -c src/trampolines.S -o $@

$(BUILD)/hook.o: src/hook.cc include/cuda_symbols.def | $(BUILD)
	$(CXX) $(CFLAGS) -c src/hook.cc -o $@

$(TARGET): $(OBJS)
	$(CXX) \
	  -shared \
	  -Wl,-soname,libcuda.so.1 \
	  -Wl,--version-script=exports.map \
		-Wl,-z,noexecstack \
		-static-libgcc -static-libstdc++ \
	  $(OBJS) \
	  -ldl -lpthread \
	  -o $(TARGET)

clean:
	rm -rf $(BUILD_DIR)
	rm -f include/cuda_symbols.def include/cuda_trampolines.inc

install: $(TARGET)
	$(SUDO) cp $(TARGET) $(LIBCUDA)

restore:
	$(SUDO) cp $(REAL_LIBCUDA) $(LIBCUDA)

.PHONY: all clean install restore