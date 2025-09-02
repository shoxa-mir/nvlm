set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cmake_minimum_required(VERSION 3.15)
project(NVLM LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define source files for the NVLM library
set(NVLM_SOURCES
    src/nvlm.cpp
    src/nvlm_impl.cpp
)

# Create shared library (DLL on Windows)
add_library(nvlm SHARED ${NVLM_SOURCES})

# Define NVLM_EXPORTS when building the DLL
target_compile_definitions(nvlm PRIVATE NVLM_EXPORTS)

set(OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../install")

# Set output directories for library
set_target_properties(nvlm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${OUTPUT_DIRECTORY}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${OUTPUT_DIRECTORY}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     "${OUTPUT_DIRECTORY}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${OUTPUT_DIRECTORY}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG          "${OUTPUT_DIRECTORY}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE        "${OUTPUT_DIRECTORY}"
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL     "${OUTPUT_DIRECTORY}"
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${OUTPUT_DIRECTORY}"
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG          "${OUTPUT_DIRECTORY}"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE        "${OUTPUT_DIRECTORY}"
    ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL     "${OUTPUT_DIRECTORY}"
    ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${OUTPUT_DIRECTORY}"
)

if(WIN32)
    # Include directories
    target_include_directories(nvlm PUBLIC 
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/third_party/onnxruntime-gpu-1.15.1/include"
        "${PROJECT_SOURCE_DIR}/third_party/opencv-4.11.0/include"
    )
    
    # Link directories
    target_link_directories(nvlm PUBLIC 
        "${PROJECT_SOURCE_DIR}/third_party/onnxruntime-gpu-1.15.1/lib/win-x64"
        "${PROJECT_SOURCE_DIR}/third_party/opencv-4.11.0/lib"
    )
    
    # Link libraries
    target_link_libraries(nvlm PUBLIC 
        "onnxruntime"
        "onnxruntime_providers_cuda"
        "onnxruntime_providers_shared"
        "onnxruntime_providers_tensorrt"
        optimized "opencv_world4110"
    )

    set(ONNX_LIB_DIR "${PROJECT_SOURCE_DIR}/third_party/onnxruntime-gpu-1.15.1/lib/win-x64")
    set(OPENCV_BIN_DIR "${PROJECT_SOURCE_DIR}/third_party/opencv-4.11.0/bin")

    set(DEPENDENCY_DLLS_TO_COPY
        "${ONNX_LIB_DIR}/onnxruntime.dll"
        "${ONNX_LIB_DIR}/onnxruntime_providers_cuda.dll"
        "${ONNX_LIB_DIR}/onnxruntime_providers_shared.dll"
        "${ONNX_LIB_DIR}/onnxruntime_providers_tensorrt.dll"
        "${OPENCV_BIN_DIR}/opencv_world4110.dll"
        "${OPENCV_BIN_DIR}/opencv_videoio_ffmpeg4110_64.dll"
    )

    # Copy dependency DLLs alongside the library
    add_custom_command(TARGET nvlm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${DEPENDENCY_DLLS_TO_COPY}
            $<TARGET_FILE_DIR:nvlm>
        COMMENT "Copying required dependency DLLs for nvlm.dll..."
    )

endif()

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Building NVLM shared library: nvlm.dll")