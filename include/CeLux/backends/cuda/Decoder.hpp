// CUDA Decoder.hpp - NVDEC Hardware Accelerated Decoder
#pragma once

#include "backends/Decoder.hpp"

#ifdef CELUX_ENABLE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace celux::backends::cuda
{

/**
 * @brief CUDA/NVDEC hardware-accelerated video decoder
 * 
 * This decoder uses FFmpeg's hwaccel API with NVDEC to decode video
 * frames directly on the GPU. The decoded NV12 frames are converted
 * to RGB using a custom CUDA kernel, and the output remains on the GPU
 * as a torch::Tensor with device='cuda'.
 * 
 * Thread safety:
 * - C++ side: Uses mutex for frame queue access
 * - CUDA side: Uses stream-ordered operations for thread safety
 */
class Decoder : public celux::Decoder
{
public:
    /**
     * @brief Construct a CUDA decoder
     * @param filePath Path to the video file
     * @param numThreads Number of CPU threads for packet demuxing
     * @param cudaDeviceIndex CUDA device index (default: 0)
     */
    Decoder(const std::string& filePath, int numThreads, int cudaDeviceIndex = 0);
    
    ~Decoder() override;
    
    // Disable copy
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;
    
    // Enable move
    Decoder(Decoder&&) noexcept;
    Decoder& operator=(Decoder&&) noexcept;
    
    /**
     * @brief Decode the next frame into the provided GPU buffer
     * @param buffer Pointer to GPU memory (must be CUDA device pointer)
     * @param frame_timestamp Optional output for frame timestamp
     * @return true if frame was decoded, false if EOF or error
     */
    bool decodeNextFrame(void* buffer, double* frame_timestamp = nullptr) override;
    
    /**
     * @brief Seek to a specific timestamp
     * @param timestamp Time in seconds
     * @return true if seek was successful
     */
    bool seek(double timestamp) override;
    
    /**
     * @brief Close the decoder and release resources
     */
    void close() override;
    
    /**
     * @brief Check if decoder is open
     */
    bool isOpen() const override;
    
    /**
     * @brief Get the CUDA device index being used
     */
    int getCudaDeviceIndex() const { return cudaDeviceIndex_; }
    
    /**
     * @brief Get the CUDA stream used for decoding
     */
    cudaStream_t getCudaStream() const { return cudaStream_; }

protected:
    void initialize(const std::string& filePath);
    void initHardwareContext();
    void initCodecContextWithHwAccel();
    
    /**
     * @brief Transfer and convert frame from NV12 to RGB on GPU
     * @param hwFrame Hardware frame from NVDEC
     * @param outputBuffer Output RGB buffer (device pointer)
     */
    void transferAndConvertFrame(AVFrame* hwFrame, void* outputBuffer);
    
    // Static callback for FFmpeg hardware pixel format selection
    static AVPixelFormat getHwFormat(AVCodecContext* ctx, const AVPixelFormat* pix_fmts);

private:
    int cudaDeviceIndex_;
    cudaStream_t cudaStream_;
    AVBufferRef* hwDeviceCtx_;
    AVPixelFormat hwPixFmt_;
    
    // Intermediate buffers for GPU processing
    void* nv12Buffer_;          // GPU buffer for NV12 data
    size_t nv12BufferSize_;
    
    // For the decoding thread
    bool hwInitialized_;
    
    // Static helper to store 'this' pointer for callback
    static thread_local Decoder* currentInstance_;
};

} // namespace celux::backends::cuda

#endif // CELUX_ENABLE_CUDA
