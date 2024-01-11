CXX = g++
NVCC = nvcc
PYTHON_INCLUDES = $(shell python3-config --includes --embed)
PYTHON_LDFLAGS = $(shell python3-config --ldflags --embed)
CXXFLAGS = -pthread -lstdc++ -std=gnu++17 -O3 -lrt -I./src -I./tests -I/usr/local/cuda/include $(PYTHON_INCLUDES)
NVCCFLAGS = -I./src -I./tests $(PYTHON_INCLUDES)
LDFLAGS = $(PYTHON_LDFLAGS)

SRC_DIR = src
TESTS_DIR = tests
BUILD_DIR = build
GPU_TESTS_DIR = gpu_tests


# Find all CUDA source files in the src folder
CU_SRCS = $(wildcard $(SRC_DIR)/*.cu)
CU_OBJS = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CU_SRCS))

# Find all test files in the tests folder
TEST_SRCS = $(wildcard $(TESTS_DIR)/*.cpp)
TEST_EXECS = $(patsubst $(TESTS_DIR)/%.cpp, $(BUILD_DIR)/%, $(TEST_SRCS))
GPU_TEST_SRCS = $(wildcard $(GPU_TESTS_DIR)/*.cu)
GPU_TEST_EXECS = $(patsubst $(GPU_TESTS_DIR)/%.cu, $(BUILD_DIR)/gpu_%, $(GPU_TEST_SRCS))

INDEX_CU = index.cu
INDEX_EXE = $(BUILD_DIR)/index

.PHONY: all clean test build_dir

all: test

test: clean build_dir $(TEST_EXECS) $(INDEX_EXE)
	@status=0; \
	for test_exec in $(TEST_EXECS) ; do \
		echo "Running $$test_exec"; \
		./$$test_exec || status=1; \
	done; \
	if [ $$status -ne 0 ]; then exit 1; fi; \
	echo "Compiling index.cu"; \
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $(INDEX_CU) -o $(INDEX_EXE)

fast_test: clean build_dir
	$(MAKE) -j $(TEST_EXECS)
	$(MAKE) -j $(GPU_TEST_EXECS)
	@status=0; \
	for test_exec in $(TEST_EXECS) ; do \
		echo "Running $$test_exec"; \
		./$$test_exec || status=1; \
	done; \
	if [ $$status -ne 0 ]; then exit 1; fi;
	echo "Compiling index.cu"; \
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $(INDEX_CU) -o $(INDEX_EXE)

gpu_fast_test: clean build_dir 
	$(MAKE) -j $(GPU_TEST_EXECS)
	@status=0; \
	for test_exec in $(GPU_TEST_EXECS) ; do \
		echo "Running $$test_exec"; \
		./$$test_exec || status=1; \
	done; \
	if [ $$status -ne 0 ]; then exit 1; fi;

$(BUILD_DIR)/gpu_%: $(GPU_TESTS_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(INDEX_EXE): $(INDEX_CU)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $< -o $@

$(BUILD_DIR)/%: $(TESTS_DIR)/%.cpp $(CU_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

build_dir:
	@mkdir -p $(BUILD_DIR)

clean:
	rm -f $(CU_OBJS) $(TEST_EXECS) $(GPU_TEST_EXECS) $(INDEX_EXE)
