# TensorRT for RTX Hello World Sample

This sample demonstrates how to use TensorRT for RTX to create, compile, and
run a simple neural network. The sample shows basic concepts such as:

- Creating a TensorRT-RTX builder and network definition.
- Building a simple fully connected neural network.
- Performing ahead-of-time (AOT) compilation.
- Configuring a runtime cache.
- Running inference with the compiled engine.

## Building the Sample

### Prerequisites

- CMake 3.10 or later
- CUDA Toolkit
- An installation of TensorRT for RTX

### Build Instructions

1. Create a build directory and navigate to it:
   ```bash
   mkdir build && cd build
   ```

2. Run CMake, pointing it to your TensorRT for RTX installation:
   ```bash
   cmake .. -DTRTRTX_INSTALL_DIR=/path/to/tensorrt-rtx
   ```

3. Build the sample:
   ```bash
   cmake --build .
   ```

## Running the Sample

After building, you can run the sample with:
```bash
./helloWorld
```

The sample will:
1. Create and compile a simple neural network.
2. Run inference with different input values.
3. Display the results.

## Code Overview

The sample demonstrates several key concepts:
- Network creation and configuration
- Engine serialization and deserialization
- Inference execution

For detailed comments explaining each step, please refer to the `helloWorld.cpp` source file. 