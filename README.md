# NVLM - Vision-Language Model Library

### This is a **computer vision/machine learning project** called NVLM that creates a shared library for Vision-Language Model processing.

## Architecture
- **C++ shared library** (DLL) that provides Vision-Language Model capabilities
- Uses **ONNX Runtime** with CUDA for GPU acceleration
- **OpenCV** for image processing
- **CMake** build system

## Core Functionality
- Load ONNX models for text or visual processing
- Encode text and images into embeddings
- Compute similarities between text and image embeddings
- Both C++ and C-style APIs for interoperability

## Key Components
- `src/nvlm.h` - Public API interface
- `src/nvlm.cpp` - API implementation wrapper
- `src/nvlm_impl.cpp` - Core implementation logic
- `third_party/` - External dependencies (ONNX Runtime, OpenCV)
- `install/` - Built library output directory

## Build Process
1. CMake configures the build with CUDA support
2. Compiles source files into shared library
3. Links with ONNX Runtime and OpenCV
4. Copies dependency DLLs to output directory

The project enables applications to perform vision-language tasks like image-text similarity matching using GPU acceleration. 
