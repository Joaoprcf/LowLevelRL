#!/bin/bash

# Check if a test name was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <test_name>"
    exit 1
fi

TEST_NAME=$1
BUILD_DIR=./build
TEST_DIR=./gpu_tests
NVCC="nvcc"
NVCCFLAGS="-I./src -I./gpu_tests"

# Create the build folder
mkdir -p build

# Compile the test
echo "Compiling $TEST_NAME..."
$NVCC $NVCCFLAGS $TEST_DIR/${TEST_NAME}_test.cu -o $BUILD_DIR/gpu_${TEST_NAME}_test

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi

# Run the test
echo "Running $TEST_NAME..."
$BUILD_DIR/gpu_${TEST_NAME}_test
