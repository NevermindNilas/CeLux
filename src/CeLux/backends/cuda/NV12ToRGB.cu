/*
 * NV12ToRGB.cu - CUDA kernels for YUV to RGB conversion
 * 
 * Inspired by NVIDIA Video Codec SDK (MIT License)
 * This implementation uses constant memory matrices and vectorized access
 * for high-performance YUV to RGB color space conversion.
 * 
 * Supports:
 * - NV12 (8-bit 4:2:0), P016 (10-bit 4:2:0)
 * - NV16 (8-bit 4:2:2), P216 (10-bit 4:2:2)
 * - YUV444 (8-bit 4:4:4), YUV444P16 (16-bit 4:4:4)
 * - Multiple color standards: BT.601, BT.709, BT.2020, FCC, SMPTE240M
 * - Limited range (16-235) and full range (0-255) support
 * - Packed RGB24 and planar RGBP output formats
 * 
 * SPDX-License-Identifier: MIT
 */

#ifdef CELUX_ENABLE_CUDA

#include <cuda_runtime.h>
#include <cstdint>

namespace celux::backends::cuda
{

//------------------------------------------------------------------------------
// Color space standards (matches FFmpeg AVCOL_SPC_* values where applicable)
//------------------------------------------------------------------------------
enum ColorSpaceStandard {
    ColorSpaceStandard_BT709 = 1,       // HD content (most common)
    ColorSpaceStandard_Unspecified = 2, // Will default to BT.709
    ColorSpaceStandard_FCC = 4,         // FCC Title 47
    ColorSpaceStandard_BT470BG = 5,     // BT.470 System B/G (PAL)
    ColorSpaceStandard_BT601 = 6,       // SD content (NTSC/PAL)
    ColorSpaceStandard_SMPTE240M = 7,   // Early HDTV
    ColorSpaceStandard_BT2020 = 9,      // HDR/UHD content
    ColorSpaceStandard_BT2020C = 10     // BT.2020 constant luminance
};

//------------------------------------------------------------------------------
// Color range
//------------------------------------------------------------------------------
enum ColorRange {
    ColorRange_Limited = 0,  // Y: 16-235, UV: 16-240 (MPEG/TV range)
    ColorRange_Full = 1      // Y: 0-255, UV: 0-255 (JPEG/PC range)
};

//------------------------------------------------------------------------------
// Constant memory for YUV to RGB conversion matrix
// Using constant memory for faster access (cached and broadcast to all threads)
//------------------------------------------------------------------------------
__constant__ float matYuv2Rgb[3][3];

//------------------------------------------------------------------------------
// RGB pixel types for vectorized access
//------------------------------------------------------------------------------
union RGB24 {
    uchar3 v;
    struct {
        uint8_t r, g, b;
    } c;
    
    __device__ __host__ RGB24() : v{0, 0, 0} {}
    __device__ __host__ RGB24(uint8_t r_, uint8_t g_, uint8_t b_) { c.r = r_; c.g = g_; c.b = b_; }
};

// For writing two RGB24 pixels at once (6 bytes, vectorized)
struct RGB24x2 {
    uchar3 x;
    uchar3 y;
};

// 32-bit RGBA with alpha
union RGBA32 {
    uint32_t d;
    uchar4 v;
    struct {
        uint8_t r, g, b, a;
    } c;
};

// For writing two RGBA32 pixels at once
struct RGBA32x2 {
    uint32_t x;
    uint32_t y;
};

//------------------------------------------------------------------------------
// Helper function to compute the YUV->RGB matrix coefficients
// Based on NVIDIA Video Codec SDK approach with extended color space support
//------------------------------------------------------------------------------
inline void GetColorSpaceConstants(int iMatrix, int colorRange, float &wr, float &wb, 
                                   int &black, int &white, int &max, int &uvMax) {
    // Set range-dependent values
    if (colorRange == ColorRange_Full) {
        black = 0;
        white = 255;
        uvMax = 255;
    } else {
        black = 16;
        white = 235;
        uvMax = 240;
    }
    max = 255;

    switch (iMatrix)
    {
    case ColorSpaceStandard_BT709:
    case ColorSpaceStandard_Unspecified:
    default:
        // BT.709 - HD content (most modern video)
        wr = 0.2126f; wb = 0.0722f;
        break;

    case ColorSpaceStandard_FCC:
        // FCC Title 47 (legacy NTSC)
        wr = 0.30f; wb = 0.11f;
        break;

    case ColorSpaceStandard_BT470BG:
    case ColorSpaceStandard_BT601:
        // BT.601 / BT.470 - SD content
        wr = 0.2990f; wb = 0.1140f;
        break;

    case ColorSpaceStandard_SMPTE240M:
        // SMPTE 240M - Early HDTV (1988-1998)
        wr = 0.212f; wb = 0.087f;
        break;

    case ColorSpaceStandard_BT2020:
    case ColorSpaceStandard_BT2020C:
        // BT.2020 - HDR/UHD content
        wr = 0.2627f; wb = 0.0593f;
        // For 10-bit content with limited range
        if (colorRange == ColorRange_Limited) {
            black = 64;   // 16 << 2 for 10-bit in 8-bit space
            white = 940 >> 2;  // Scaled for 8-bit processing
        }
        break;
    }
}

//------------------------------------------------------------------------------
// Set the YUV to RGB conversion matrix in constant memory
//------------------------------------------------------------------------------
void SetMatYuv2Rgb(int iMatrix, int colorRange, cudaStream_t stream) {
    float wr, wb;
    int black, white, max, uvMax;
    GetColorSpaceConstants(iMatrix, colorRange, wr, wb, black, white, max, uvMax);
    
    // Compute conversion matrix coefficients
    // Based on the standard YUV to RGB equations:
    // R = Y + (1-wr)/0.5 * V
    // G = Y - wb*(1-wb)/(0.5*(1-wb-wr)) * U - wr*(1-wr)/(0.5*(1-wb-wr)) * V
    // B = Y + (1-wb)/0.5 * U
    float mat[3][3] = {
        {1.0f, 0.0f, (1.0f - wr) / 0.5f},
        {1.0f, -wb * (1.0f - wb) / 0.5f / (1 - wb - wr), -wr * (1 - wr) / 0.5f / (1 - wb - wr)},
        {1.0f, (1.0f - wb) / 0.5f, 0.0f},
    };
    
    // Scale for range conversion
    float yScale = static_cast<float>(max) / static_cast<float>(white - black);
    float uvScale = static_cast<float>(max) / static_cast<float>(uvMax - black);
    
    for (int i = 0; i < 3; i++) {
        mat[i][0] *= yScale;      // Y coefficient
        mat[i][1] *= uvScale;     // U coefficient
        mat[i][2] *= uvScale;     // V coefficient
    }
    
    cudaMemcpyToSymbolAsync(matYuv2Rgb, mat, sizeof(mat), 0, cudaMemcpyHostToDevice, stream);
}

// Overload for backwards compatibility (defaults to limited range)
void SetMatYuv2Rgb(int iMatrix, cudaStream_t stream) {
    SetMatYuv2Rgb(iMatrix, ColorRange_Limited, stream);
}

//------------------------------------------------------------------------------
// Device helper functions
//------------------------------------------------------------------------------
template<class T>
__device__ __forceinline__ T Clamp(T x, T lower, T upper) {
    return x < lower ? lower : (x > upper ? upper : x);
}

/**
 * @brief Convert a single YUV pixel to RGB using the constant memory matrix
 * 
 * @tparam YuvUnit Type of YUV component (uint8_t for 8-bit, uint16_t for 10/16-bit)
 * @param y Y component
 * @param u U component  
 * @param v V component
 * @param fullRange Whether input is full range (0-255) or limited (16-235)
 * @return RGB24 pixel
 */
template<class YuvUnit>
__device__ __forceinline__ RGB24 YuvToRgbForPixel(YuvUnit y, YuvUnit u, YuvUnit v, bool fullRange = false) {
    // Calculate offsets based on bit depth
    const int bitDepth = sizeof(YuvUnit) * 8;
    const int low = fullRange ? 0 : (1 << (bitDepth - 4));   // Y offset: 0 or 16 for 8-bit
    const int mid = 1 << (bitDepth - 1);                      // UV offset: 128 for 8-bit
    const float maxf = static_cast<float>((1 << bitDepth) - 1);
    
    float fy = static_cast<float>(static_cast<int>(y) - low);
    float fu = static_cast<float>(static_cast<int>(u) - mid);
    float fv = static_cast<float>(static_cast<int>(v) - mid);
    
    // Apply YUV to RGB matrix multiplication
    float rf = matYuv2Rgb[0][0] * fy + matYuv2Rgb[0][1] * fu + matYuv2Rgb[0][2] * fv;
    float gf = matYuv2Rgb[1][0] * fy + matYuv2Rgb[1][1] * fu + matYuv2Rgb[1][2] * fv;
    float bf = matYuv2Rgb[2][0] * fy + matYuv2Rgb[2][1] * fu + matYuv2Rgb[2][2] * fv;
    
    // For high bit depth, scale down to 8-bit
    if (bitDepth > 8) {
        const float scale = 255.0f / maxf;
        rf *= scale;
        gf *= scale;
        bf *= scale;
    }
    
    RGB24 rgb;
    rgb.c.r = static_cast<uint8_t>(Clamp(rf, 0.0f, 255.0f));
    rgb.c.g = static_cast<uint8_t>(Clamp(gf, 0.0f, 255.0f));
    rgb.c.b = static_cast<uint8_t>(Clamp(bf, 0.0f, 255.0f));
    return rgb;
}

/**
 * @brief Convert YUV to RGBA32 with alpha channel
 */
template<class YuvUnit>
__device__ __forceinline__ RGBA32 YuvToRgbaForPixel(YuvUnit y, YuvUnit u, YuvUnit v, uint8_t alpha = 255, bool fullRange = false) {
    RGB24 rgb = YuvToRgbForPixel(y, u, v, fullRange);
    RGBA32 rgba;
    rgba.c.r = rgb.c.r;
    rgba.c.g = rgb.c.g;
    rgba.c.b = rgb.c.b;
    rgba.c.a = alpha;
    return rgba;
}

//==============================================================================
// NV12 KERNELS (4:2:0, 8-bit)
//==============================================================================

/**
 * @brief NV12 to RGB24 kernel - processes 2x2 pixel blocks with vectorized writes
 */
__global__ void Nv12ToRgb24Kernel(
    const uint8_t* __restrict__ pNv12,
    int nNv12Pitch,
    uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    int nWidth,
    int nSurfaceHeight,
    int nHeight,
    bool fullRange)
{
    // Each thread processes a 2x2 block of pixels
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }
    
    // Vectorized Y reads (2 pixels at once)
    const uint8_t* pSrcY = pNv12 + y * nNv12Pitch + x;
    uchar2 y0 = *reinterpret_cast<const uchar2*>(pSrcY);
    uchar2 y1 = *reinterpret_cast<const uchar2*>(pSrcY + nNv12Pitch);
    
    // Read UV pair (shared by 2x2 block)
    const uint8_t* pSrcUV = pNv12 + nSurfaceHeight * nNv12Pitch + (y / 2) * nNv12Pitch + x;
    uchar2 uv = *reinterpret_cast<const uchar2*>(pSrcUV);
    
    // Convert 4 pixels
    RGB24 rgb00 = YuvToRgbForPixel<uint8_t>(y0.x, uv.x, uv.y, fullRange);
    RGB24 rgb01 = YuvToRgbForPixel<uint8_t>(y0.y, uv.x, uv.y, fullRange);
    RGB24 rgb10 = YuvToRgbForPixel<uint8_t>(y1.x, uv.x, uv.y, fullRange);
    RGB24 rgb11 = YuvToRgbForPixel<uint8_t>(y1.y, uv.x, uv.y, fullRange);
    
    // Vectorized RGB writes - write 2 pixels per row at once
    RGB24x2* pDst0 = reinterpret_cast<RGB24x2*>(pRgb + y * nRgbPitch + x * 3);
    RGB24x2* pDst1 = reinterpret_cast<RGB24x2*>(pRgb + (y + 1) * nRgbPitch + x * 3);
    
    *pDst0 = RGB24x2{rgb00.v, rgb01.v};
    *pDst1 = RGB24x2{rgb10.v, rgb11.v};
}

/**
 * @brief NV12 to planar RGB (RGBP) kernel for ML workflows
 * Output: R plane [0:H*W], G plane [H*W:2*H*W], B plane [2*H*W:3*H*W]
 */
__global__ void Nv12ToRgbPlanarKernel(
    const uint8_t* __restrict__ pNv12,
    int nNv12Pitch,
    uint8_t* __restrict__ pRgbp,
    int nWidth,
    int nSurfaceHeight,
    int nHeight,
    bool fullRange)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }
    
    const int planeSize = nWidth * nHeight;
    
    // Read Y values
    const uint8_t* pSrcY = pNv12 + y * nNv12Pitch + x;
    uchar2 y0 = *reinterpret_cast<const uchar2*>(pSrcY);
    uchar2 y1 = *reinterpret_cast<const uchar2*>(pSrcY + nNv12Pitch);
    
    // Read UV
    const uint8_t* pSrcUV = pNv12 + nSurfaceHeight * nNv12Pitch + (y / 2) * nNv12Pitch + x;
    uchar2 uv = *reinterpret_cast<const uchar2*>(pSrcUV);
    
    // Convert
    RGB24 rgb00 = YuvToRgbForPixel<uint8_t>(y0.x, uv.x, uv.y, fullRange);
    RGB24 rgb01 = YuvToRgbForPixel<uint8_t>(y0.y, uv.x, uv.y, fullRange);
    RGB24 rgb10 = YuvToRgbForPixel<uint8_t>(y1.x, uv.x, uv.y, fullRange);
    RGB24 rgb11 = YuvToRgbForPixel<uint8_t>(y1.y, uv.x, uv.y, fullRange);
    
    // Write to planar format - R plane
    int idx0 = y * nWidth + x;
    int idx1 = (y + 1) * nWidth + x;
    
    // Vectorized writes to each plane
    *reinterpret_cast<uchar2*>(pRgbp + idx0) = make_uchar2(rgb00.c.r, rgb01.c.r);
    *reinterpret_cast<uchar2*>(pRgbp + idx1) = make_uchar2(rgb10.c.r, rgb11.c.r);
    
    // G plane
    *reinterpret_cast<uchar2*>(pRgbp + planeSize + idx0) = make_uchar2(rgb00.c.g, rgb01.c.g);
    *reinterpret_cast<uchar2*>(pRgbp + planeSize + idx1) = make_uchar2(rgb10.c.g, rgb11.c.g);
    
    // B plane
    *reinterpret_cast<uchar2*>(pRgbp + 2 * planeSize + idx0) = make_uchar2(rgb00.c.b, rgb01.c.b);
    *reinterpret_cast<uchar2*>(pRgbp + 2 * planeSize + idx1) = make_uchar2(rgb10.c.b, rgb11.c.b);
}

/**
 * @brief NV12 to RGB24 with separate Y and UV plane pointers
 */
__global__ void Nv12SeparateToRgb24Kernel(
    const uint8_t* __restrict__ pY,
    const uint8_t* __restrict__ pUV,
    int nYPitch,
    int nUVPitch,
    uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    bool fullRange)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }
    
    // Read Y
    const uint8_t* pSrcY = pY + y * nYPitch + x;
    uchar2 y0 = *reinterpret_cast<const uchar2*>(pSrcY);
    uchar2 y1 = *reinterpret_cast<const uchar2*>(pSrcY + nYPitch);
    
    // Read UV
    const uint8_t* pSrcUV = pUV + (y / 2) * nUVPitch + x;
    uchar2 uv = *reinterpret_cast<const uchar2*>(pSrcUV);
    
    // Convert
    RGB24 rgb00 = YuvToRgbForPixel<uint8_t>(y0.x, uv.x, uv.y, fullRange);
    RGB24 rgb01 = YuvToRgbForPixel<uint8_t>(y0.y, uv.x, uv.y, fullRange);
    RGB24 rgb10 = YuvToRgbForPixel<uint8_t>(y1.x, uv.x, uv.y, fullRange);
    RGB24 rgb11 = YuvToRgbForPixel<uint8_t>(y1.y, uv.x, uv.y, fullRange);
    
    // Vectorized writes
    RGB24x2* pDst0 = reinterpret_cast<RGB24x2*>(pRgb + y * nRgbPitch + x * 3);
    RGB24x2* pDst1 = reinterpret_cast<RGB24x2*>(pRgb + (y + 1) * nRgbPitch + x * 3);
    
    *pDst0 = RGB24x2{rgb00.v, rgb01.v};
    *pDst1 = RGB24x2{rgb10.v, rgb11.v};
}

//==============================================================================
// P016 KERNELS (4:2:0, 10/16-bit)
//==============================================================================

/**
 * @brief P016 (10-bit NV12) to RGB24 kernel for HDR content
 */
__global__ void P016ToRgb24Kernel(
    const uint8_t* __restrict__ pP016,
    int nP016Pitch,
    uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    int nWidth,
    int nSurfaceHeight,
    int nHeight,
    bool fullRange)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }
    
    // Read 16-bit Y values
    const uint16_t* pSrcY0 = reinterpret_cast<const uint16_t*>(pP016 + y * nP016Pitch) + x;
    const uint16_t* pSrcY1 = reinterpret_cast<const uint16_t*>(pP016 + (y + 1) * nP016Pitch) + x;
    ushort2 y0 = *reinterpret_cast<const ushort2*>(pSrcY0);
    ushort2 y1 = *reinterpret_cast<const ushort2*>(pSrcY1);
    
    // Read 16-bit UV
    const uint16_t* pSrcUV = reinterpret_cast<const uint16_t*>(pP016 + nSurfaceHeight * nP016Pitch + (y / 2) * nP016Pitch) + x;
    ushort2 uv = *reinterpret_cast<const ushort2*>(pSrcUV);
    
    // Convert
    RGB24 rgb00 = YuvToRgbForPixel<uint16_t>(y0.x, uv.x, uv.y, fullRange);
    RGB24 rgb01 = YuvToRgbForPixel<uint16_t>(y0.y, uv.x, uv.y, fullRange);
    RGB24 rgb10 = YuvToRgbForPixel<uint16_t>(y1.x, uv.x, uv.y, fullRange);
    RGB24 rgb11 = YuvToRgbForPixel<uint16_t>(y1.y, uv.x, uv.y, fullRange);
    
    // Vectorized writes
    RGB24x2* pDst0 = reinterpret_cast<RGB24x2*>(pRgb + y * nRgbPitch + x * 3);
    RGB24x2* pDst1 = reinterpret_cast<RGB24x2*>(pRgb + (y + 1) * nRgbPitch + x * 3);
    
    *pDst0 = RGB24x2{rgb00.v, rgb01.v};
    *pDst1 = RGB24x2{rgb10.v, rgb11.v};
}

/**
 * @brief P016 to planar RGB for ML workflows
 */
__global__ void P016ToRgbPlanarKernel(
    const uint8_t* __restrict__ pP016,
    int nP016Pitch,
    uint8_t* __restrict__ pRgbp,
    int nWidth,
    int nSurfaceHeight,
    int nHeight,
    bool fullRange)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }
    
    const int planeSize = nWidth * nHeight;
    
    // Read 16-bit Y values
    const uint16_t* pSrcY0 = reinterpret_cast<const uint16_t*>(pP016 + y * nP016Pitch) + x;
    const uint16_t* pSrcY1 = reinterpret_cast<const uint16_t*>(pP016 + (y + 1) * nP016Pitch) + x;
    ushort2 y0 = *reinterpret_cast<const ushort2*>(pSrcY0);
    ushort2 y1 = *reinterpret_cast<const ushort2*>(pSrcY1);
    
    // Read 16-bit UV
    const uint16_t* pSrcUV = reinterpret_cast<const uint16_t*>(pP016 + nSurfaceHeight * nP016Pitch + (y / 2) * nP016Pitch) + x;
    ushort2 uv = *reinterpret_cast<const ushort2*>(pSrcUV);
    
    // Convert
    RGB24 rgb00 = YuvToRgbForPixel<uint16_t>(y0.x, uv.x, uv.y, fullRange);
    RGB24 rgb01 = YuvToRgbForPixel<uint16_t>(y0.y, uv.x, uv.y, fullRange);
    RGB24 rgb10 = YuvToRgbForPixel<uint16_t>(y1.x, uv.x, uv.y, fullRange);
    RGB24 rgb11 = YuvToRgbForPixel<uint16_t>(y1.y, uv.x, uv.y, fullRange);
    
    // Write to planar format
    int idx0 = y * nWidth + x;
    int idx1 = (y + 1) * nWidth + x;
    
    *reinterpret_cast<uchar2*>(pRgbp + idx0) = make_uchar2(rgb00.c.r, rgb01.c.r);
    *reinterpret_cast<uchar2*>(pRgbp + idx1) = make_uchar2(rgb10.c.r, rgb11.c.r);
    *reinterpret_cast<uchar2*>(pRgbp + planeSize + idx0) = make_uchar2(rgb00.c.g, rgb01.c.g);
    *reinterpret_cast<uchar2*>(pRgbp + planeSize + idx1) = make_uchar2(rgb10.c.g, rgb11.c.g);
    *reinterpret_cast<uchar2*>(pRgbp + 2 * planeSize + idx0) = make_uchar2(rgb00.c.b, rgb01.c.b);
    *reinterpret_cast<uchar2*>(pRgbp + 2 * planeSize + idx1) = make_uchar2(rgb10.c.b, rgb11.c.b);
}

//==============================================================================
// NV16 KERNELS (4:2:2, 8-bit) - Professional video
//==============================================================================

/**
 * @brief NV16 (4:2:2) to RGB24 kernel
 * UV has same height as Y but half width
 */
__global__ void Nv16ToRgb24Kernel(
    const uint8_t* __restrict__ pNv16,
    int nNv16Pitch,
    uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    int nWidth,
    int nSurfaceHeight,
    int nHeight,
    bool fullRange)
{
    // Each thread processes 2 horizontal pixels (they share UV)
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }
    
    // Read 2 Y values
    const uint8_t* pSrcY = pNv16 + y * nNv16Pitch + x;
    uchar2 yy = *reinterpret_cast<const uchar2*>(pSrcY);
    
    // Read UV (same row, half width) - UV plane starts at nSurfaceHeight
    const uint8_t* pSrcUV = pNv16 + nSurfaceHeight * nNv16Pitch + y * nNv16Pitch + x;
    uchar2 uv = *reinterpret_cast<const uchar2*>(pSrcUV);
    
    // Convert 2 pixels
    RGB24 rgb0 = YuvToRgbForPixel<uint8_t>(yy.x, uv.x, uv.y, fullRange);
    RGB24 rgb1 = YuvToRgbForPixel<uint8_t>(yy.y, uv.x, uv.y, fullRange);
    
    // Vectorized write
    RGB24x2* pDst = reinterpret_cast<RGB24x2*>(pRgb + y * nRgbPitch + x * 3);
    *pDst = RGB24x2{rgb0.v, rgb1.v};
}

/**
 * @brief NV16 to planar RGB
 */
__global__ void Nv16ToRgbPlanarKernel(
    const uint8_t* __restrict__ pNv16,
    int nNv16Pitch,
    uint8_t* __restrict__ pRgbp,
    int nWidth,
    int nSurfaceHeight,
    int nHeight,
    bool fullRange)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }
    
    const int planeSize = nWidth * nHeight;
    
    const uint8_t* pSrcY = pNv16 + y * nNv16Pitch + x;
    uchar2 yy = *reinterpret_cast<const uchar2*>(pSrcY);
    
    const uint8_t* pSrcUV = pNv16 + nSurfaceHeight * nNv16Pitch + y * nNv16Pitch + x;
    uchar2 uv = *reinterpret_cast<const uchar2*>(pSrcUV);
    
    RGB24 rgb0 = YuvToRgbForPixel<uint8_t>(yy.x, uv.x, uv.y, fullRange);
    RGB24 rgb1 = YuvToRgbForPixel<uint8_t>(yy.y, uv.x, uv.y, fullRange);
    
    int idx = y * nWidth + x;
    *reinterpret_cast<uchar2*>(pRgbp + idx) = make_uchar2(rgb0.c.r, rgb1.c.r);
    *reinterpret_cast<uchar2*>(pRgbp + planeSize + idx) = make_uchar2(rgb0.c.g, rgb1.c.g);
    *reinterpret_cast<uchar2*>(pRgbp + 2 * planeSize + idx) = make_uchar2(rgb0.c.b, rgb1.c.b);
}

//==============================================================================
// P216 KERNELS (4:2:2, 10/16-bit) - Professional HDR video
//==============================================================================

/**
 * @brief P216 (10-bit 4:2:2) to RGB24 kernel
 */
__global__ void P216ToRgb24Kernel(
    const uint8_t* __restrict__ pP216,
    int nP216Pitch,
    uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    int nWidth,
    int nSurfaceHeight,
    int nHeight,
    bool fullRange)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }
    
    // Read 16-bit Y values
    const uint16_t* pSrcY = reinterpret_cast<const uint16_t*>(pP216 + y * nP216Pitch) + x;
    ushort2 yy = *reinterpret_cast<const ushort2*>(pSrcY);
    
    // Read 16-bit UV
    const uint16_t* pSrcUV = reinterpret_cast<const uint16_t*>(pP216 + nSurfaceHeight * nP216Pitch + y * nP216Pitch) + x;
    ushort2 uv = *reinterpret_cast<const ushort2*>(pSrcUV);
    
    // Convert
    RGB24 rgb0 = YuvToRgbForPixel<uint16_t>(yy.x, uv.x, uv.y, fullRange);
    RGB24 rgb1 = YuvToRgbForPixel<uint16_t>(yy.y, uv.x, uv.y, fullRange);
    
    // Vectorized write
    RGB24x2* pDst = reinterpret_cast<RGB24x2*>(pRgb + y * nRgbPitch + x * 3);
    *pDst = RGB24x2{rgb0.v, rgb1.v};
}

//==============================================================================
// YUV444 KERNELS (4:4:4, 8-bit) - High quality, no chroma subsampling
//==============================================================================

/**
 * @brief YUV444 planar to RGB24 kernel
 * Each pixel has its own U and V values
 */
__global__ void Yuv444ToRgb24Kernel(
    const uint8_t* __restrict__ pY,
    const uint8_t* __restrict__ pU,
    const uint8_t* __restrict__ pV,
    int nYuvPitch,
    uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    bool fullRange)
{
    // Each thread processes 2 horizontal pixels
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }
    
    // Read Y, U, V (each pixel has its own values)
    const uint8_t* pSrcY = pY + y * nYuvPitch + x;
    const uint8_t* pSrcU = pU + y * nYuvPitch + x;
    const uint8_t* pSrcV = pV + y * nYuvPitch + x;
    
    uchar2 yy = *reinterpret_cast<const uchar2*>(pSrcY);
    uchar2 uu = *reinterpret_cast<const uchar2*>(pSrcU);
    uchar2 vv = *reinterpret_cast<const uchar2*>(pSrcV);
    
    // Convert (each pixel gets its own U, V)
    RGB24 rgb0 = YuvToRgbForPixel<uint8_t>(yy.x, uu.x, vv.x, fullRange);
    RGB24 rgb1 = YuvToRgbForPixel<uint8_t>(yy.y, uu.y, vv.y, fullRange);
    
    // Vectorized write
    RGB24x2* pDst = reinterpret_cast<RGB24x2*>(pRgb + y * nRgbPitch + x * 3);
    *pDst = RGB24x2{rgb0.v, rgb1.v};
}

/**
 * @brief YUV444 planar to planar RGB
 */
__global__ void Yuv444ToRgbPlanarKernel(
    const uint8_t* __restrict__ pY,
    const uint8_t* __restrict__ pU,
    const uint8_t* __restrict__ pV,
    int nYuvPitch,
    uint8_t* __restrict__ pRgbp,
    int nWidth,
    int nHeight,
    bool fullRange)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }
    
    const int planeSize = nWidth * nHeight;
    
    const uint8_t* pSrcY = pY + y * nYuvPitch + x;
    const uint8_t* pSrcU = pU + y * nYuvPitch + x;
    const uint8_t* pSrcV = pV + y * nYuvPitch + x;
    
    uchar2 yy = *reinterpret_cast<const uchar2*>(pSrcY);
    uchar2 uu = *reinterpret_cast<const uchar2*>(pSrcU);
    uchar2 vv = *reinterpret_cast<const uchar2*>(pSrcV);
    
    RGB24 rgb0 = YuvToRgbForPixel<uint8_t>(yy.x, uu.x, vv.x, fullRange);
    RGB24 rgb1 = YuvToRgbForPixel<uint8_t>(yy.y, uu.y, vv.y, fullRange);
    
    int idx = y * nWidth + x;
    *reinterpret_cast<uchar2*>(pRgbp + idx) = make_uchar2(rgb0.c.r, rgb1.c.r);
    *reinterpret_cast<uchar2*>(pRgbp + planeSize + idx) = make_uchar2(rgb0.c.g, rgb1.c.g);
    *reinterpret_cast<uchar2*>(pRgbp + 2 * planeSize + idx) = make_uchar2(rgb0.c.b, rgb1.c.b);
}

//==============================================================================
// YUV444P16 KERNELS (4:4:4, 16-bit) - Professional HDR
//==============================================================================

/**
 * @brief YUV444 16-bit planar to RGB24 kernel
 */
__global__ void Yuv444P16ToRgb24Kernel(
    const uint8_t* __restrict__ pY,
    const uint8_t* __restrict__ pU,
    const uint8_t* __restrict__ pV,
    int nYuvPitch,
    uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    bool fullRange)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }
    
    // Read 16-bit Y, U, V
    const uint16_t* pSrcY = reinterpret_cast<const uint16_t*>(pY + y * nYuvPitch) + x;
    const uint16_t* pSrcU = reinterpret_cast<const uint16_t*>(pU + y * nYuvPitch) + x;
    const uint16_t* pSrcV = reinterpret_cast<const uint16_t*>(pV + y * nYuvPitch) + x;
    
    ushort2 yy = *reinterpret_cast<const ushort2*>(pSrcY);
    ushort2 uu = *reinterpret_cast<const ushort2*>(pSrcU);
    ushort2 vv = *reinterpret_cast<const ushort2*>(pSrcV);
    
    // Convert
    RGB24 rgb0 = YuvToRgbForPixel<uint16_t>(yy.x, uu.x, vv.x, fullRange);
    RGB24 rgb1 = YuvToRgbForPixel<uint16_t>(yy.y, uu.y, vv.y, fullRange);
    
    // Vectorized write
    RGB24x2* pDst = reinterpret_cast<RGB24x2*>(pRgb + y * nRgbPitch + x * 3);
    *pDst = RGB24x2{rgb0.v, rgb1.v};
}

//==============================================================================
// PUBLIC API FUNCTIONS
//==============================================================================

/**
 * @brief Initialize color space conversion matrix
 * 
 * @param colorSpace Color space standard (1=BT.709, 6=BT.601, 9=BT.2020, etc.)
 * @param colorRange 0=Limited (TV), 1=Full (PC/JPEG)
 * @param stream CUDA stream
 */
void initColorSpaceMatrix(int colorSpace, int colorRange, cudaStream_t stream) {
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
}

// Backwards compatible overload
void initColorSpaceMatrix(int colorSpace, cudaStream_t stream) {
    SetMatYuv2Rgb(colorSpace, ColorRange_Limited, stream);
}

//------------------------------------------------------------------------------
// NV12 (4:2:0, 8-bit) Launch Functions
//------------------------------------------------------------------------------

/**
 * @brief Convert NV12 to RGB24 (packed)
 */
void launchNv12ToRgb24(
    const uint8_t* pNv12,
    int nNv12Pitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 2);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    Nv12ToRgb24Kernel<<<gridDim, blockDim, 0, stream>>>(
        pNv12, nNv12Pitch, pRgb, nRgbPitch, nWidth, nHeight, nHeight, 
        colorRange == ColorRange_Full
    );
}

/**
 * @brief Convert NV12 to planar RGB (RGBP) for ML workflows
 * Output layout: [R plane][G plane][B plane], each H*W bytes
 */
void launchNv12ToRgbPlanar(
    const uint8_t* pNv12,
    int nNv12Pitch,
    uint8_t* pRgbp,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 2);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    Nv12ToRgbPlanarKernel<<<gridDim, blockDim, 0, stream>>>(
        pNv12, nNv12Pitch, pRgbp, nWidth, nHeight, nHeight,
        colorRange == ColorRange_Full
    );
}

/**
 * @brief Convert NV12 to RGB24 (separate Y and UV planes)
 */
void launchNv12ToRgb24Separate(
    const uint8_t* pY,
    const uint8_t* pUV,
    int nYPitch,
    int nUVPitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 2);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    Nv12SeparateToRgb24Kernel<<<gridDim, blockDim, 0, stream>>>(
        pY, pUV, nYPitch, nUVPitch, pRgb, nRgbPitch, nWidth, nHeight,
        colorRange == ColorRange_Full
    );
}

//------------------------------------------------------------------------------
// P016 (4:2:0, 10/16-bit) Launch Functions
//------------------------------------------------------------------------------

/**
 * @brief Convert P016 (10-bit NV12) to RGB24
 */
void launchP016ToRgb24(
    const uint8_t* pP016,
    int nP016Pitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 2);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    P016ToRgb24Kernel<<<gridDim, blockDim, 0, stream>>>(
        pP016, nP016Pitch, pRgb, nRgbPitch, nWidth, nHeight, nHeight,
        colorRange == ColorRange_Full
    );
}

/**
 * @brief Convert P016 to planar RGB
 */
void launchP016ToRgbPlanar(
    const uint8_t* pP016,
    int nP016Pitch,
    uint8_t* pRgbp,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 2);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    P016ToRgbPlanarKernel<<<gridDim, blockDim, 0, stream>>>(
        pP016, nP016Pitch, pRgbp, nWidth, nHeight, nHeight,
        colorRange == ColorRange_Full
    );
}

//------------------------------------------------------------------------------
// NV16 (4:2:2, 8-bit) Launch Functions
//------------------------------------------------------------------------------

/**
 * @brief Convert NV16 (4:2:2) to RGB24
 */
void launchNv16ToRgb24(
    const uint8_t* pNv16,
    int nNv16Pitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    // 4:2:2 - UV same height as Y, so only horizontal 2x1 processing
    dim3 blockDim(32, 4);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    Nv16ToRgb24Kernel<<<gridDim, blockDim, 0, stream>>>(
        pNv16, nNv16Pitch, pRgb, nRgbPitch, nWidth, nHeight, nHeight,
        colorRange == ColorRange_Full
    );
}

/**
 * @brief Convert NV16 to planar RGB
 */
void launchNv16ToRgbPlanar(
    const uint8_t* pNv16,
    int nNv16Pitch,
    uint8_t* pRgbp,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 4);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    Nv16ToRgbPlanarKernel<<<gridDim, blockDim, 0, stream>>>(
        pNv16, nNv16Pitch, pRgbp, nWidth, nHeight, nHeight,
        colorRange == ColorRange_Full
    );
}

//------------------------------------------------------------------------------
// P216 (4:2:2, 10/16-bit) Launch Functions
//------------------------------------------------------------------------------

/**
 * @brief Convert P216 (10-bit 4:2:2) to RGB24
 */
void launchP216ToRgb24(
    const uint8_t* pP216,
    int nP216Pitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 4);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    P216ToRgb24Kernel<<<gridDim, blockDim, 0, stream>>>(
        pP216, nP216Pitch, pRgb, nRgbPitch, nWidth, nHeight, nHeight,
        colorRange == ColorRange_Full
    );
}

//------------------------------------------------------------------------------
// YUV444 (4:4:4, 8-bit) Launch Functions
//------------------------------------------------------------------------------

/**
 * @brief Convert YUV444 planar to RGB24
 */
void launchYuv444ToRgb24(
    const uint8_t* pY,
    const uint8_t* pU,
    const uint8_t* pV,
    int nYuvPitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 4);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    Yuv444ToRgb24Kernel<<<gridDim, blockDim, 0, stream>>>(
        pY, pU, pV, nYuvPitch, pRgb, nRgbPitch, nWidth, nHeight,
        colorRange == ColorRange_Full
    );
}

/**
 * @brief Convert YUV444 planar to planar RGB
 */
void launchYuv444ToRgbPlanar(
    const uint8_t* pY,
    const uint8_t* pU,
    const uint8_t* pV,
    int nYuvPitch,
    uint8_t* pRgbp,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 4);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    Yuv444ToRgbPlanarKernel<<<gridDim, blockDim, 0, stream>>>(
        pY, pU, pV, nYuvPitch, pRgbp, nWidth, nHeight,
        colorRange == ColorRange_Full
    );
}

//------------------------------------------------------------------------------
// YUV444P16 (4:4:4, 16-bit) Launch Functions
//------------------------------------------------------------------------------

/**
 * @brief Convert YUV444 16-bit planar to RGB24
 */
void launchYuv444P16ToRgb24(
    const uint8_t* pY,
    const uint8_t* pU,
    const uint8_t* pV,
    int nYuvPitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 4);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    Yuv444P16ToRgb24Kernel<<<gridDim, blockDim, 0, stream>>>(
        pY, pU, pV, nYuvPitch, pRgb, nRgbPitch, nWidth, nHeight,
        colorRange == ColorRange_Full
    );
}

//==============================================================================
// LEGACY API COMPATIBILITY
// These functions maintain backwards compatibility with existing code
//==============================================================================

void launchNV12ToRGBKernel(
    const uint8_t* yPlane,
    uint8_t* rgbOutput,
    int width,
    int height,
    int yPitch,
    int rgbPitch,
    cudaStream_t stream)
{
    // Default to BT.709 for HD content, limited range
    launchNv12ToRgb24(yPlane, yPitch, rgbOutput, rgbPitch, width, height,
                      ColorSpaceStandard_BT709, ColorRange_Limited, stream);
}

void launchNV12ToRGBKernelWithUV(
    const uint8_t* yPlane,
    const uint8_t* uvPlane,
    uint8_t* rgbOutput,
    int width,
    int height,
    int yPitch,
    int uvPitch,
    int rgbPitch,
    cudaStream_t stream)
{
    launchNv12ToRgb24Separate(yPlane, uvPlane, yPitch, uvPitch, rgbOutput, rgbPitch,
                              width, height, ColorSpaceStandard_BT709, ColorRange_Limited, stream);
}

// Overloads without colorRange parameter (for backwards compatibility)
void launchNv12ToRgb24(
    const uint8_t* pNv12,
    int nNv12Pitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    cudaStream_t stream)
{
    launchNv12ToRgb24(pNv12, nNv12Pitch, pRgb, nRgbPitch, nWidth, nHeight,
                      colorSpace, ColorRange_Limited, stream);
}

void launchP016ToRgb24(
    const uint8_t* pP016,
    int nP016Pitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    cudaStream_t stream)
{
    launchP016ToRgb24(pP016, nP016Pitch, pRgb, nRgbPitch, nWidth, nHeight,
                      colorSpace, ColorRange_Limited, stream);
}

} // namespace celux::backends::cuda

#endif // CELUX_ENABLE_CUDA
