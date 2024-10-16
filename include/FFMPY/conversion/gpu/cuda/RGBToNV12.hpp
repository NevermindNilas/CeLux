// RGBToNV12.hpp

#pragma once

#include "Frame.hpp"
#include "BaseConverter.hpp"

extern "C"
{
                     
    void rgb_to_nv12(const unsigned char* rgbInput, int width, int height,
                     int rgbStride, unsigned char* yPlane, unsigned char* uvPlane,
                     int yStride, int uvStride, cudaStream_t stream = 0);
    void rgb_to_nv12_float(const float* rgbInput, int width, int height, int rgbStride,
                           unsigned char* yPlane, unsigned char* uvPlane, int yStride,
                           int uvStride, cudaStream_t stream = 0);
    void rgb_to_nv12_half(const __half* rgbInput, int width, int height, int rgbStride,
                          unsigned char* yPlane, unsigned char* uvPlane, int yStride,
                          int uvStride, cudaStream_t stream = 0);
}

namespace ffmpy
{
namespace conversion
{
namespace gpu
{
namespace cuda
{

template <typename T> class RGBToNV12 : public ConverterBase<T>
{
  public:
    RGBToNV12();
    RGBToNV12(cudaStream_t stream);
    ~RGBToNV12();

    void convert(ffmpy::Frame& frame, void* buffer) override;
};

// Template Definitions

template <typename T> RGBToNV12<T>::RGBToNV12() : ConverterBase<T>()
{
}

template <typename T>
RGBToNV12<T>::RGBToNV12(cudaStream_t stream) : ConverterBase<T>(stream)
{
}

template <typename T> RGBToNV12<T>::~RGBToNV12()
{
}

template <typename T> void RGBToNV12<T>::convert(ffmpy::Frame& frame, void* buffer)
{
    const unsigned char* rgbInput = static_cast<const unsigned char*>(buffer);
    unsigned char* yPlane = frame.getData(0);
    unsigned char* uvPlane = frame.getData(1);
    int yStride = frame.getLineSize(0);
    int uvStride = frame.getLineSize(1);
    int width = frame.getWidth();
    int height = frame.getHeight();
    int rgbStride = width * 3;

    if constexpr (std::is_same<T, uint8_t>::value)
    {
        // Call the kernel for uint8_t
        rgb_to_nv12(rgbInput, width, height, rgbStride, yPlane, uvPlane, yStride,
                    uvStride, this->conversionStream);
    }
    else if constexpr (std::is_same<T, float>::value)
    {
        // Call the kernel for float
        rgb_to_nv12_float(reinterpret_cast<const float*>(rgbInput), width, height,
                          rgbStride, yPlane, uvPlane, yStride, uvStride,
                          this->conversionStream);
    }
    else if constexpr (std::is_same<T, __half>::value)
    {
        // Call the kernel for __half
        rgb_to_nv12_half(reinterpret_cast<const __half*>(rgbInput), width, height,
                         rgbStride, yPlane, uvPlane, yStride, uvStride,
                         this->conversionStream);
    }
    else
    {
        static_assert(sizeof(T) == 0, "Unsupported data type for RGBToNV12");
    }

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CUDA kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

} // namespace cuda
} // namespace gpu
} // namespace conversion
} // namespace ffmpy
