/*
 * RGBToAutoGPU.hpp - GPU-based RGB to YUV conversion for encoding
 * 
 * This is the GPU equivalent of RGBToAutoLibyuv.hpp, providing
 * CUDA-accelerated color space conversion for the NVENC pipeline.
 * 
 * SPDX-License-Identifier: MIT
 */

#pragma once

#ifdef CELUX_ENABLE_CUDA

#include <CeLux/backends/cuda/RGBToNV12.cuh>
#include <cuda_runtime.h>
#include <stdexcept>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
}

namespace celux::conversion::gpu
{

/**
 * @brief GPU-based RGB24 to YUV converter for encoding
 * 
 * Supports conversion from GPU RGB24 tensors to various YUV formats
 * for direct upload to NVENC hardware encoder.
 */
class RGBToAutoGPUConverter
{
private:
    int width;
    int height;
    AVPixelFormat dst_fmt;
    cudaStream_t stream;
    
    // Color space settings (default to BT.709 limited for HD content)
    int colorSpace = celux::backends::cuda::ColorSpaceEncode_BT709;
    int colorRange = celux::backends::cuda::ColorRangeEncode_Limited;

public:
    RGBToAutoGPUConverter(int w, int h, AVPixelFormat format, cudaStream_t cudaStream = nullptr)
        : width(w), height(h), dst_fmt(format), stream(cudaStream)
    {
        // Validate supported formats
        switch (format) {
            case AV_PIX_FMT_NV12:
            case AV_PIX_FMT_P010LE:
            case AV_PIX_FMT_YUV444P:
            case AV_PIX_FMT_NV16:
                break;
            default:
                throw std::runtime_error("RGBToAutoGPUConverter: Unsupported pixel format");
        }
    }
    
    /**
     * @brief Set color space for conversion
     * @param space Color space standard (BT.601, BT.709, BT.2020)
     */
    void setColorSpace(int space) {
        colorSpace = space;
    }
    
    /**
     * @brief Set color range for conversion
     * @param range Color range (Limited or Full)
     */
    void setColorRange(int range) {
        colorRange = range;
    }
    
    /**
     * @brief Convert GPU RGB24 buffer to YUV format in AVFrame
     * 
     * @param frame Destination AVFrame (must be allocated with av_frame_get_buffer)
     * @param gpuRgb Pointer to RGB24 data in GPU memory (HWC format, 3 bytes per pixel)
     * @param rgbPitch Pitch of RGB buffer in bytes (typically width * 3)
     */
    void convert(AVFrame* frame, const uint8_t* gpuRgb, int rgbPitch = 0)
    {
        if (!frame || !gpuRgb) {
            throw std::runtime_error("RGBToAutoGPUConverter::convert: null pointer");
        }
        
        if (rgbPitch == 0) {
            rgbPitch = width * 3;  // Default: packed RGB24
        }
        
        switch (dst_fmt) {
            case AV_PIX_FMT_NV12:
                convertToNV12(frame, gpuRgb, rgbPitch);
                break;
                
            case AV_PIX_FMT_P010LE:
                convertToP010(frame, gpuRgb, rgbPitch);
                break;
                
            case AV_PIX_FMT_YUV444P:
                convertToYUV444P(frame, gpuRgb, rgbPitch);
                break;
                
            case AV_PIX_FMT_NV16:
                convertToNV16(frame, gpuRgb, rgbPitch);
                break;
                
            default:
                throw std::runtime_error("RGBToAutoGPUConverter: Unsupported format");
        }
    }

private:
    void convertToNV12(AVFrame* frame, const uint8_t* gpuRgb, int rgbPitch)
    {
        // NV12: Y plane at data[0], UV plane at data[1]
        // For semi-planar NV12, surface height = height (Y plane)
        celux::backends::cuda::Rgb24ToNv12(
            gpuRgb,
            rgbPitch,
            frame->data[0],
            frame->linesize[0],
            width,
            height,
            height,  // Surface height for Y plane
            colorSpace,
            colorRange,
            stream
        );
    }
    
    void convertToP010(AVFrame* frame, const uint8_t* gpuRgb, int rgbPitch)
    {
        celux::backends::cuda::Rgb24ToP010(
            gpuRgb,
            rgbPitch,
            frame->data[0],
            frame->linesize[0],
            width,
            height,
            height,
            colorSpace,
            colorRange,
            stream
        );
    }
    
    void convertToYUV444P(AVFrame* frame, const uint8_t* gpuRgb, int rgbPitch)
    {
        celux::backends::cuda::Rgb24ToYuv444(
            gpuRgb,
            rgbPitch,
            frame->data[0],  // Y
            frame->data[1],  // U
            frame->data[2],  // V
            frame->linesize[0],
            width,
            height,
            colorSpace,
            colorRange,
            stream
        );
    }
    
    void convertToNV16(AVFrame* frame, const uint8_t* gpuRgb, int rgbPitch)
    {
        celux::backends::cuda::Rgb24ToNv16(
            gpuRgb,
            rgbPitch,
            frame->data[0],
            frame->linesize[0],
            width,
            height,
            height,
            colorSpace,
            colorRange,
            stream
        );
    }
};

} // namespace celux::conversion::gpu

#endif // CELUX_ENABLE_CUDA
