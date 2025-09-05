#include "nvlm_impl.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#ifdef _WIN32
#include <windows.h>
#endif

namespace nvlm {

    NVLMImpl::NVLMImpl() 
        : env_(ORT_LOGGING_LEVEL_WARNING, "NVLM"),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        
        // Initialize session options with optimizations
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        std::cout << "[NVLM] Initialized with ONNX Runtime" << std::endl;
    }

    NVLMImpl::~NVLMImpl() {
        session_.reset();
        std::cout << "[NVLM] Cleanup completed" << std::endl;
    }

    bool NVLMImpl::LoadModel(const std::string& model_path, 
                            ProcessingMode mode, 
                            const std::string& model_name) {
        
        std::lock_guard<std::mutex> lock(session_mutex_);
        
        try {
            std::cout << "[NVLM] Loading model: " << model_path << std::endl;
            std::cout << "[NVLM] Model name: " << model_name << std::endl;
            std::cout << "[NVLM] Processing mode: " << (mode == ProcessingMode::Visual ? "Visual" : "Textual") << std::endl;
            
            // Store model information
            model_path_ = model_path;
            model_name_ = model_name;
            current_mode_ = mode;
            
            // Setup CUDA execution provider
            if (!SetupCudaProvider()) {
                SetError("Failed to setup CUDA provider");
                return false;
            }
            
            // Convert string path to wide string for Windows
            #ifdef _WIN32
                std::wstring wide_path(model_path.begin(), model_path.end());
                
                // Create session
                session_ = std::make_unique<Ort::Session>(env_, wide_path.c_str(), session_options_);
            #else
                session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
            #endif
            
            // Query model input/output information
            QueryModelInfo();
            
            std::cout << "[NVLM] Model loaded successfully!" << std::endl;
            return true;
            
        } catch (const Ort::Exception& e) {
            SetError("ONNX Runtime error: " + std::string(e.what()));
            session_.reset();
            return false;
        } catch (const std::exception& e) {
            SetError("Standard error: " + std::string(e.what()));
            session_.reset();
            return false;
        }
    }

    bool NVLMImpl::SetupCudaProvider() {
        try {
            std::cout << "[NVLM] Setting up CUDA execution provider..." << std::endl;
            
            // Static flag to prevent multiple CUDA provider registrations
            static bool cuda_provider_added = false;
            if (cuda_provider_added) {
                std::cout << "[NVLM] CUDA provider already configured, skipping..." << std::endl;
                return true;
            }
            
            // CUDA Provider options
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;  // Use GPU 0
            cuda_options.arena_extend_strategy = 0;  // kNextPowerOfTwo
            cuda_options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;  // 2GB limit
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.do_copy_in_default_stream = 1;
            
            // Add CUDA provider
            session_options_.AppendExecutionProvider_CUDA(cuda_options);
            cuda_provider_added = true;
            
            // Also add CPU provider as fallback
            // Note: CPU provider is automatically added as the last provider
            
            std::cout << "[NVLM] CUDA provider configured successfully" << std::endl;
            return true;
            
        } catch (const Ort::Exception& e) {
            std::cout << "[NVLM] CUDA setup failed: " << e.what() << std::endl;
            std::cout << "[NVLM] Falling back to CPU execution" << std::endl;
            
            // Clear any existing providers and use CPU only
            session_options_ = Ort::SessionOptions{};
            session_options_.SetIntraOpNumThreads(1);
            session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            
            return true;  // CPU fallback is still success
        }
    }

    void NVLMImpl::QueryModelInfo() {
        if (!session_) {
            SetError("No session available for querying model info");
            return;
        }
        
        try {
            Ort::AllocatorWithDefaultOptions allocator;
            
            // Clear previous information
            input_names_.clear();
            output_names_.clear();
            input_shapes_.clear();
            output_shapes_.clear();
            
            // Get input information
            size_t num_inputs = session_->GetInputCount();
            std::cout << "[NVLM] Model has " << num_inputs << " input(s)" << std::endl;
            
            for (size_t i = 0; i < num_inputs; i++) {
                // Get input name
                auto input_name = session_->GetInputNameAllocated(i, allocator);
                input_names_.push_back(std::string(input_name.get()));
                
                // Get input type info
                Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                
                // Get input shape
                std::vector<int64_t> input_shape = tensor_info.GetShape();
                input_shapes_.push_back(input_shape);
                
                std::cout << "[NVLM] Input " << i << ": " << input_names_[i] << " - Shape: [";
                for (size_t j = 0; j < input_shape.size(); j++) {
                    std::cout << input_shape[j];
                    if (j < input_shape.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
            
            // Get output information
            size_t num_outputs = session_->GetOutputCount();
            std::cout << "[NVLM] Model has " << num_outputs << " output(s)" << std::endl;
            
            for (size_t i = 0; i < num_outputs; i++) {
                // Get output name
                auto output_name = session_->GetOutputNameAllocated(i, allocator);
                output_names_.push_back(std::string(output_name.get()));
                
                // Get output type info
                Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                
                // Get output shape
                std::vector<int64_t> output_shape = tensor_info.GetShape();
                output_shapes_.push_back(output_shape);
                
                std::cout << "[NVLM] Output " << i << ": " << output_names_[i] << " - Shape: [";
                for (size_t j = 0; j < output_shape.size(); j++) {
                    std::cout << output_shape[j];
                    if (j < output_shape.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
            
        } catch (const Ort::Exception& e) {
            SetError("Error querying model info: " + std::string(e.what()));
        }
    }

    void NVLMImpl::SetError(const std::string& error) {
        last_error_ = error;
        std::cerr << "[NVLM ERROR] " << error << std::endl;
    }

    // Placeholder implementations for other functions (we'll implement these next)
    std::vector<float> NVLMImpl::PreprocessText(const std::string& text) {
        SetError("PreprocessText not implemented yet");
        return {};
    }

    std::vector<float> NVLMImpl::PreprocessImage(const std::vector<uint8_t>& image_data, 
                                               int width, int height, int channels) {
        try {
            // CLIP preprocessing constants
            const int target_size = 224;
            const std::vector<float> mean = {0.48145466f, 0.4578275f, 0.40821073f}; // RGB
            const std::vector<float> std = {0.26862954f, 0.26130258f, 0.27577711f};  // RGB
            
            std::cout << "[NVLM] Preprocessing image: " << width << "x" << height 
                      << " with " << channels << " channels" << std::endl;
            
            // Validate input
            if (channels != 3) {
                SetError("Image must have exactly 3 channels (RGB)");
                return {};
            }
            
            if (image_data.size() != static_cast<size_t>(width * height * channels)) {
                SetError("Image data size doesn't match dimensions");
                return {};
            }
            
            // Create OpenCV Mat from input data (assuming BGR order from OpenCV)
            cv::Mat img(height, width, CV_8UC3, const_cast<uint8_t*>(image_data.data()));
            
            // Convert BGR to RGB
            cv::Mat rgb_img;
            cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
            
            // Resize to 224x224 (CLIP standard)
            cv::Mat resized_img;
            cv::resize(rgb_img, resized_img, cv::Size(target_size, target_size), 0, 0, cv::INTER_LINEAR);
            
            // Convert to float and normalize to [0, 1]
            cv::Mat float_img;
            resized_img.convertTo(float_img, CV_32F, 1.0 / 255.0);
            
            // Prepare output tensor in CHW format: [C, H, W]
            std::vector<float> preprocessed_data(3 * target_size * target_size);
            
            // Apply normalization and convert to CHW format
            for (int c = 0; c < 3; ++c) {
                for (int h = 0; h < target_size; ++h) {
                    for (int w = 0; w < target_size; ++w) {
                        // Get pixel value from HWC format
                        float pixel_value = float_img.at<cv::Vec3f>(h, w)[c];
                        
                        // Normalize with ImageNet statistics
                        float normalized_value = (pixel_value - mean[c]) / std[c];
                        
                        // Store in CHW format
                        int chw_index = c * target_size * target_size + h * target_size + w;
                        preprocessed_data[chw_index] = normalized_value;
                    }
                }
            }
            
            std::cout << "[NVLM] Image preprocessing completed: " << preprocessed_data.size() 
                      << " float values" << std::endl;
            
            return preprocessed_data;
            
        } catch (const cv::Exception& e) {
            SetError("OpenCV error during image preprocessing: " + std::string(e.what()));
            return {};
        } catch (const std::exception& e) {
            SetError("Error during image preprocessing: " + std::string(e.what()));
            return {};
        }
    }

    Embedding NVLMImpl::EncodeText(const std::string& text) {
        SetError("EncodeText not implemented yet");
        return Embedding(512);  // Placeholder dimension
    }

    Embedding NVLMImpl::EncodeImage(const std::vector<uint8_t>& image_data, 
                                  int width, int height, int channels) {
        try {
            std::lock_guard<std::mutex> lock(session_mutex_);
            
            if (!session_) {
                SetError("No model loaded for image encoding");
                return Embedding(512);
            }
            
            std::cout << "[NVLM] Encoding image: " << width << "x" << height 
                      << " with " << channels << " channels" << std::endl;
            
            // Step 1: Preprocess the image
            std::vector<float> preprocessed_data = PreprocessImage(image_data, width, height, channels);
            if (preprocessed_data.empty()) {
                SetError("Failed to preprocess image for encoding");
                return Embedding(512);
            }
            
            // Step 2: Create input tensor
            const int64_t batch_size = 1;
            const int64_t num_channels = 3;
            const int64_t img_height = 224;
            const int64_t img_width = 224;
            
            std::vector<int64_t> input_shape{batch_size, num_channels, img_height, img_width};
            
            // Create the input tensor
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, 
                preprocessed_data.data(), 
                preprocessed_data.size(), 
                input_shape.data(), 
                input_shape.size()
            );
            
            // Prepare input and output names
            std::vector<const char*> input_names_cstr;
            std::vector<const char*> output_names_cstr;
            
            for (const auto& name : input_names_) {
                input_names_cstr.push_back(name.c_str());
            }
            for (const auto& name : output_names_) {
                output_names_cstr.push_back(name.c_str());
            }
            
            // Step 3: Run inference
            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(std::move(input_tensor));
            
            std::cout << "[NVLM] Running image encoding inference..." << std::endl;
            
            auto output_tensors = session_->Run(
                Ort::RunOptions{nullptr}, 
                input_names_cstr.data(), 
                input_tensors.data(), 
                input_tensors.size(), 
                output_names_cstr.data(), 
                output_names_cstr.size()
            );
            
            if (output_tensors.empty()) {
                SetError("No output from image encoding model");
                return Embedding(512);
            }
            
            // Step 4: Extract embedding from output tensor
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            const auto& output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            
            // Calculate total number of elements in the output
            size_t total_elements = 1;
            for (int64_t dim : output_shape) {
                total_elements *= dim;
            }
            
            std::cout << "[NVLM] Image encoding completed. Embedding dimension: " << total_elements << std::endl;
            
            // Create embedding result
            Embedding result(total_elements);
            std::copy(output_data, output_data + total_elements, result.data.begin());
            
            return result;
            
        } catch (const Ort::Exception& e) {
            SetError("ONNX Runtime error in image encoding: " + std::string(e.what()));
            return Embedding(512);
        } catch (const std::exception& e) {
            SetError("Error in image encoding: " + std::string(e.what()));
            return Embedding(512);
        }
    }

    SimilarityResult NVLMImpl::ComputeSimilarity(const Embedding& text_emb, 
                                               const Embedding& image_emb) {
        SetError("ComputeSimilarity not implemented yet");
        return {0.0f};
    }
}

