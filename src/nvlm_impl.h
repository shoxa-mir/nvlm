#pragma once

#include "nvlm.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <mutex>

namespace nvlm {
    class NVLMImpl {
    private:
        // ONNX Runtime components
        Ort::Env env_;
        Ort::SessionOptions session_options_;
        std::unique_ptr<Ort::Session> session_;
        Ort::MemoryInfo memory_info_;
        
        // Model information
        std::string model_path_;
        std::string model_name_;
        ProcessingMode current_mode_;
        
        // Input/Output information
        std::vector<std::string> input_names_;
        std::vector<std::string> output_names_;
        std::vector<std::vector<int64_t>> input_shapes_;
        std::vector<std::vector<int64_t>> output_shapes_;
        
        // Thread safety
        std::mutex session_mutex_;
        
        // Error handling
        std::string last_error_;

    public:
        NVLMImpl();
        ~NVLMImpl();
        
        // Core functionality
        bool LoadModel(const std::string& model_path, 
                      ProcessingMode mode, 
                      const std::string& model_name);
        
        std::vector<float> PreprocessText(const std::string& text);
        std::vector<float> PreprocessImage(const std::vector<uint8_t>& image_data, 
                                         int width, int height, int channels);
        
        Embedding EncodeText(const std::string& text);
        Embedding EncodeImage(const std::vector<uint8_t>& image_data, 
                            int width, int height, int channels);
        
        SimilarityResult ComputeSimilarity(const Embedding& text_emb, 
                                         const Embedding& image_emb);
        
        // Utility functions
        bool IsModelLoaded() const { return session_ != nullptr; }
        std::string GetLastError() const { return last_error_; }
        
    private:
        // Helper functions
        bool SetupCudaProvider();
        void QueryModelInfo();
        void SetError(const std::string& error);
    };
}