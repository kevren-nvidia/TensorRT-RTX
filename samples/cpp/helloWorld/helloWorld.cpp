/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "NvInfer.h"
#include "NvInferRuntime.h"

#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <vector>

//! TensorRT-RTX applications are responsible for implementing the
//! nvinfer1::ILogger interface. This is used to log messages from the
//! TensorRT-RTX library.
class Logger : public nvinfer1::ILogger
{
public:
    Logger() = default;
    ~Logger() override = default;
private:
    std::string severityToString(nvinfer1::ILogger::Severity severity) {
        switch (severity) {
            case nvinfer1::ILogger::Severity::kVERBOSE: return "VERBOSE";
            case nvinfer1::ILogger::Severity::kINFO: return "INFO";
            case nvinfer1::ILogger::Severity::kWARNING: return "WARNING";
            case nvinfer1::ILogger::Severity::kERROR: return "ERROR";
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
            default: return "UNKNOWN";
        }
    }

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        std::cout << severityToString(severity) << ": " << msg << std::endl;
    }
};

//! Create a builder configuration. This is used to configure options for
//! how you want your network to be optimized.
std::unique_ptr<nvinfer1::IBuilderConfig> createBuilderConfig(
    nvinfer1::IBuilder& builder, Logger& logger)
{
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder.createBuilderConfig());

    // In this example, we intend to run ahead-of-time (AOT) compilation on
    // the end-user's machine, so we set the compute capability to kCURRENT.
    // This provides the fastest ahead-of-time compilation, but produces an
    // engine that is only compatible with the current GPU.
    //
    // For engines that are deployed with the application to a diverse set of
    // GPUs, you should leave the compute capability unset. The default
    // behavior is to support all RTX compute capabilities, Ampere and later.
    config->setNbComputeCapabilities(1);
    config->setComputeCapability(nvinfer1::ComputeCapability::kCURRENT, /* index */ 0);

    return config;
}

// These sizes are arbitrary.
constexpr int32_t kInputSize = 3;
constexpr int32_t kHiddenSize = 10;
constexpr int32_t kOutputSize = 2;

struct WeightsData {
    // The weights in this example are initialized to 1.0f, but typically would
    // be loaded from a file or other source.
    WeightsData() : fc1WeightsData(kInputSize * kHiddenSize, 1.0f),
                    fc2WeightsData(kHiddenSize * kOutputSize, 1.0f) {}

    std::vector<float> fc1WeightsData;
    std::vector<float> fc2WeightsData;
};

//! Create a simple fully connected network with one input, one hidden layer, and one output.
std::unique_ptr<nvinfer1::INetworkDefinition> createNetwork(nvinfer1::IBuilder& builder, const WeightsData& weightsData)
{
    // TensorRT-RTX only supports strongly typed networks; if we don't explicitly
    // set this, we get a warning log.
    nvinfer1::NetworkDefinitionCreationFlags flags =
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder.createNetworkV2(flags));

    // Add network input
    auto input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{1, kInputSize});

    // Add a fully connected layer for the inputs and the hidden layer.
    nvinfer1::Weights fc1Weights = nvinfer1::Weights{nvinfer1::DataType::kFLOAT,
                                                     weightsData.fc1WeightsData.data(),
                                                     static_cast<int64_t>(weightsData.fc1WeightsData.size())};
    auto fc1WeightsLayer = network->addConstant(nvinfer1::Dims2{kInputSize, kHiddenSize}, fc1Weights);
    fc1WeightsLayer->setName("fully connected layer 1 weights");
    auto fc1 = network->addMatrixMultiply(*input, nvinfer1::MatrixOperation::kNONE,
                                          *fc1WeightsLayer->getOutput(0), nvinfer1::MatrixOperation::kNONE);
    fc1->setName("fully connected layer 1");    
    auto relu = network->addActivation(*fc1->getOutput(0), nvinfer1::ActivationType::kRELU);
    relu->setName("relu activation");
    
    // Add a fully connected layer for the hidden layer and the output.
    nvinfer1::Weights fc2Weights = nvinfer1::Weights{nvinfer1::DataType::kFLOAT,
                                                     weightsData.fc2WeightsData.data(),
                                                     static_cast<int64_t>(weightsData.fc2WeightsData.size())};
    auto fc2WeightsLayer = network->addConstant(nvinfer1::Dims2{kHiddenSize, kOutputSize}, fc2Weights);
    fc2WeightsLayer->setName("fully connected layer 2 weights");
    auto fc2 = network->addMatrixMultiply(*relu->getOutput(0), nvinfer1::MatrixOperation::kNONE,
                                          *fc2WeightsLayer->getOutput(0), nvinfer1::MatrixOperation::kNONE);
    fc2->setName("fully connected layer 2");
    
    // Mark the output tensor.
    fc2->getOutput(0)->setName("output tensor");
    network->markOutput(*fc2->getOutput(0));

    return network;
 }

//! Build the serialized engine.
//! In TensorRT-RTX, we often refer to this stage as "Ahead-of-Time" (AOT)
//! compilation. This stage tends to be slower than the "Just-in-Time" (JIT)
//! compilation stage. For this reason, you should perform this operation at
//! installation time or first run, and then save the resulting engine.
//!
//! You may choose to build the engine once and then deploy it to end-users;
//! it is OS-independent and by default supports Ampere and later GPUs. But
//! be aware that the engine does not guarantee forward compatibility, so
//! you must build a new engine for each new TensorRT-RTX version.
std::unique_ptr<nvinfer1::IHostMemory> createSerializedEngine()
{
    Logger logger;
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger));
    auto config = createBuilderConfig(*builder, logger);

    // The data backing IConstantLayers must remain valid until the engine has
    // been built; therefore we create weightsData here.
    WeightsData weightsData;
    auto network = createNetwork(*builder, weightsData);
    
    std::unique_ptr<nvinfer1::IHostMemory> serializedEngine(
        builder->buildSerializedNetwork(*network, *config));

    return serializedEngine;
}

class InferenceContext
{
public:
    InferenceContext(const nvinfer1::IHostMemory& serializedEngine);
    ~InferenceContext();

    std::vector<float> runInference(const std::vector<float>& input) const;

    bool isValid() const {
        return mRuntime && mEngine && mContext && mRuntimeConfig
            && mRuntimeCache;
    }

    std::unique_ptr<nvinfer1::IHostMemory> serializeRuntimeCache() const
    {
        assert(isValid());
        return std::unique_ptr<nvinfer1::IHostMemory>(
            mRuntimeCache->serialize());
    }

private:
    Logger mLogger;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IRuntimeConfig> mRuntimeConfig;
    std::unique_ptr<nvinfer1::IRuntimeCache> mRuntimeCache;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    std::vector<void*> mBindings;
};

#define CUDA_ASSERT(cudaCall) \
    do { \
        cudaError_t __cudaError = (cudaCall); \
        if (__cudaError != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(__cudaError) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            assert(false); \
        } \
    } while (0)

InferenceContext::InferenceContext(const nvinfer1::IHostMemory& serializedEngine)
: mRuntime(nvinfer1::createInferRuntime(mLogger))
, mEngine(mRuntime->deserializeCudaEngine(serializedEngine.data(), serializedEngine.size()))
{
    if (!mEngine) {
        std::cerr << "Failed to create CUDA engine!" << std::endl;
        return;
    }

    // The IRuntimeConfig allows you to configure the behavior of the inference
    // runtime.
    mRuntimeConfig = std::unique_ptr<nvinfer1::IRuntimeConfig>(mEngine->createRuntimeConfig());
    if (!mRuntimeConfig) {
        std::cerr << "Failed to create runtime config!" << std::endl;
        return;
    }

    // TensorRT-RTX supports a runtime cache which you may save to persistent
    // storage and reload on future runs of your application.
    mRuntimeCache.reset(mRuntimeConfig->createRuntimeCache());
    mRuntimeConfig->setRuntimeCache(*mRuntimeCache);

    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext(mRuntimeConfig.get()));
    if (!mContext) {
        std::cerr << "Failed to create execution context!" << std::endl;
        return;
    }

    mBindings.resize(2);
    CUDA_ASSERT(cudaMalloc(&mBindings[0], kInputSize * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&mBindings[1], kOutputSize * sizeof(float)));
}

InferenceContext::~InferenceContext()
{
    CUDA_ASSERT(cudaFree(mBindings[0]));
    CUDA_ASSERT(cudaFree(mBindings[1]));
}

std::vector<float> InferenceContext::runInference(const std::vector<float>& input) const
{
    CUDA_ASSERT(cudaMemcpy(mBindings[0], input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    mContext->executeV2(mBindings.data());

    std::vector<float> output(kOutputSize);
    CUDA_ASSERT(cudaMemcpy(output.data(), mBindings[1], output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    return output;
}

template <typename T>
void dumpContainer(std::ostream& os, const std::string& name, const T& container) {
    os << name << ": ";
    for (const auto& value : container) {
        os << value << " ";
    }
    os << std::endl;
}

int main() {
    std::unique_ptr<nvinfer1::IHostMemory> serializedEngine =
        createSerializedEngine();
    if (!serializedEngine) {
        std::cerr << "Failed to build serialized engine!" << std::endl;
        return 1;
    }
    std::cout << "Successfully built the network. Engine size: "
              << serializedEngine->size() << " bytes." << std::endl;

    InferenceContext context(*serializedEngine);
    if (!context.isValid()) {   
        std::cerr << "Failed to create inference context!" << std::endl;
        return 1;
    }
    
    for (int i = 0; i < 5; i++) {
        std::vector<float> input(kInputSize, static_cast<float>(i));
        std::vector<float> output = context.runInference(input);
        dumpContainer(std::cout, "Input", input);
        dumpContainer(std::cout, "Output", output);
    }

    std::cout << "Successfully ran the network." << std::endl;

    // Now that we have finished running inference and we want to shut down our
    // application, we can serialize the runtime cache. Normally here we would
    // save the serialized cache to persistent storage.
    std::unique_ptr<nvinfer1::IHostMemory> serializedCache =
        context.serializeRuntimeCache();
    if (!serializedCache) {
        std::cerr << "Failed to serialize runtime cache!" << std::endl;
        return 1;
    }
    std::cout << "Successfully serialized the runtime cache. Cache size: "
              << serializedCache->size() << " bytes." << std::endl;

    return 0;
}