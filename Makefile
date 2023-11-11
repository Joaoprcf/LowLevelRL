CXX = g++
NVCC = nvcc
CXXFLAGS = -pthread -lstdc++ -std=gnu++17 -lrt -ldl -lm -I./src -I./tests
NVCCFLAGS = -I./src -I./tests
LDFLAGS =

SRC_DIR = src
TESTS_DIR = tests
BUILD_DIR = build

# Find all CUDA source files in the src folder
CU_SRCS = $(wildcard $(SRC_DIR)/*.cu)
CU_OBJS = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CU_SRCS))

# Find all test files in the tests folder
TEST_SRCS = $(wildcard $(TESTS_DIR)/*.cpp)
TEST_EXECS = $(patsubst $(TESTS_DIR)/%.cpp, $(BUILD_DIR)/%, $(TEST_SRCS))

.PHONY: all clean test build_dir

all: test

test: clean build_dir $(TEST_EXECS)
	@for test_exec in $(TEST_EXECS) ; do \
		echo "Running $$test_exec"; \
		./$$test_exec; \
	done

$(BUILD_DIR)/%: $(TESTS_DIR)/%.cpp $(CU_OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

build_dir:
	@mkdir -p $(BUILD_DIR)

clean:
	rm -f $(CU_OBJS) $(TEST_EXECS)
