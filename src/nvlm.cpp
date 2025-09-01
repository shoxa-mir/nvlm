#include "nvlm.h"
#include "nvlm_impl.h"

namespace nvlm {

    // NVLM class implementation
    NVLM::NVLM() : impl_(std::make_unique<NVLMImpl>()) {}
    
    NVLM::~NVLM() = default;
    
    NVLM::NVLM(NVLM&&) noexcept = default;
    NVLM& NVLM::operator=(NVLM&&) noexcept = default;

    bool NVLM::LoadModel(const std::string& model_path, 
                        ProcessingMode mode, 
                        const std::string& model_name) {
        return impl_->LoadModel(model_path, mode, model_name);
    }

    std::vector<float> NVLM::PreprocessText(const std::string& text) {
        return impl_->PreprocessText(text);
    }

    std::vector<float> NVLM::PreprocessImage(const std::vector<uint8_t>& image_data, 
                                           int width, int height, int channels) {
        return impl_->PreprocessImage(image_data, width, height, channels);
    }

    Embedding NVLM::EncodeText(const std::string& text) {
        return impl_->EncodeText(text);
    }

    Embedding NVLM::EncodeImage(const std::vector<uint8_t>& image_data, 
                              int width, int height, int channels) {
        return impl_->EncodeImage(image_data, width, height, channels);
    }

    SimilarityResult NVLM::ComputeSimilarity(const Embedding& text_emb, 
                                           const Embedding& image_emb) {
        return impl_->ComputeSimilarity(text_emb, image_emb);
    }

    bool NVLM::IsModelLoaded() const {
        return impl_->IsModelLoaded();
    }

    std::string NVLM::GetLastError() const {
        return impl_->GetLastError();
    }
}

// C-style wrapper function implementations
extern "C" {
    
    // Static storage for last error message (thread-local would be better for production)
    static thread_local std::string g_last_error;
    
    NVLM_API void* CreateNVLMInstance() {
        try {
            return new nvlm::NVLM();
        } catch (const std::exception& e) {
            g_last_error = e.what();
            return nullptr;
        }
    }
    
    NVLM_API void DeleteNVLMInstance(void* instance) {
        if (instance) {
            delete static_cast<nvlm::NVLM*>(instance);
        }
    }
    
    NVLM_API bool NVLM_LoadModel(void* instance, const char* model_path, int mode, const char* model_name) {
        if (!instance || !model_path || !model_name) {
            g_last_error = "Invalid parameters";
            return false;
        }
        
        try {
            auto* nvlm = static_cast<nvlm::NVLM*>(instance);
            auto processing_mode = static_cast<nvlm::ProcessingMode>(mode);
            bool result = nvlm->LoadModel(model_path, processing_mode, model_name);
            if (!result) {
                g_last_error = nvlm->GetLastError();
            }
            return result;
        } catch (const std::exception& e) {
            g_last_error = e.what();
            return false;
        }
    }
    
    NVLM_API bool NVLM_IsModelLoaded(void* instance) {
        if (!instance) return false;
        
        try {
            auto* nvlm = static_cast<nvlm::NVLM*>(instance);
            return nvlm->IsModelLoaded();
        } catch (...) {
            return false;
        }
    }
    
    NVLM_API const char* NVLM_GetLastError(void* instance) {
        if (instance) {
            try {
                auto* nvlm = static_cast<nvlm::NVLM*>(instance);
                g_last_error = nvlm->GetLastError();
                return g_last_error.c_str();
            } catch (...) {
                g_last_error = "Exception in NVLM_GetLastError";
            }
        }
        return g_last_error.c_str();
    }
}