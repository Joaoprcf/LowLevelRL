# Experimental optimized c++ neuralnetowrk graph

## Run the main file

```bash
nvcc --std c++17 -I./src index.cu -o index.o && ./index.o
```

## Run tests sequentially

```bash
make test
```

## Run tests in parallel

```bash
make fast_test
```

## Test singular test

This script will compile and execute test/example_test.cpp

```bash
bash test_singular.sh example
```

## Build docker file

```bash
docker build . -t lowlevelrl
```

## Run docker file

```bash
docker run lowlevelrl
```
