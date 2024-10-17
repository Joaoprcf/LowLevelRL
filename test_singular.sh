#!/bin/bash

# Check if at least a test name was provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <test_name> [debug]"
    exit 1
fi

TEST_NAME=$1
BUILD_DIR=./build
TEST_DIR=./tests
CXX=g++
CXXFLAGS="-pthread -static-libstdc++ -std=gnu++17 -O3 -lrt -ldl -lm -I./src -I./tests -I/usr/local/cuda/include"
LDFLAGS=

# Check for the debug flag
if [ "$#" -eq 2 ] && [ "$2" = "debug" ]; then
    CXXFLAGS="$CXXFLAGS -O0 -g -fsanitize=address"
    echo "Debug mode enabled"
fi

# Create the build folder
mkdir -p build

rm -f $BUILD_DIR/${TEST_NAME}_test

# Compile the test
echo "Compiling $TEST_NAME..."
$CXX $CXXFLAGS $TEST_DIR/${TEST_NAME}_test.cpp -o $BUILD_DIR/${TEST_NAME}_test $LDFLAGS

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi

# Run the test
echo "Running $TEST_NAME..."
$BUILD_DIR/${TEST_NAME}_test 
