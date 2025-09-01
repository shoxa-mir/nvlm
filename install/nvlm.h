#pragma once

#include <string>
#include <vector>
#include <memory>

// Define import macros for DLL usage
#ifndef NVLM_EXPORTS
    #define NVLM_API __declspec(dllimport)
#else
    #define NVLM_API __declspec(dllexport)
#endif

namespace nvlm {

    // Enum for processing modes
    enum class ProcessingMode {
        Visual,
        Textual
    };

    // Structure to hold embedding results
    struct Embedding {
        std::vector<float> data;
        size_t dimensions;
        
        Embedding(size_t dim) : dimensions(dim) {
            data.resize(dim);
        }
    };

    // Structure for similarity results
    struct SimilarityResult {
        float cosine_similarity;
        // Add more similarity metrics as needed
    };

    // Forward declaration of implementation class
    class NVLMImpl;

    // Main NVLM class - this is what users will interact with
    class NVLM_API NVLM {
    private:
        std::unique_ptr<NVLMImpl> impl_;

    public:
        NVLM();
        ~NVLM();

        // Disable copy constructor and assignment (we'll use move semantics)
        NVLM(const NVLM&) = delete;
        NVLM& operator=(const NVLM&) = delete;
        NVLM(NVLM&&) noexcept;
        NVLM& operator=(NVLM&&) noexcept;

        // 1. Load ONNX model with CUDA
        bool LoadModel(const std::string& model_path, 
                      ProcessingMode mode, 
                      const std::string& model_name);

        // 2. Preprocess input data
        std::vector<float> PreprocessText(const std::string& text);
        std::vector<float> PreprocessImage(const std::vector<uint8_t>& image_data, 
                                         int width, int height, int channels);

        // 3. Encode data and return embeddings
        Embedding EncodeText(const std::string& text);
        Embedding EncodeImage(const std::vector<uint8_t>& image_data, 
                            int width, int height, int channels);

        // 4. Compute similarities
        SimilarityResult ComputeSimilarity(const Embedding& text_emb, 
                                         const Embedding& image_emb);
        
        // Utility functions
        bool IsModelLoaded() const;
        std::string GetLastError() const;
    };
}