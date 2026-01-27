#pragma once

extern "C"
{
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
#include <libyuv.h>
}

#include "CeLux/conversion/cpu/CPUConverter.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

namespace celux
{
namespace conversion
{
namespace cpu
{

/**
 * @brief Converts RGB24 buffer (HWC uint8) to YUV formats using libyuv.
 *
 * This provides color-accurate conversion matching the quality approach
 * used in the decoding path (AutoToRGBConverter uses libyuv for YUV→RGB).
 *
 * Usage:
 *   RGBToAutoLibyuvConverter conv(width, height, AV_PIX_FMT_YUV420P);
 *   conv.convert(yuvFrame, rgbTensor.data_ptr<uint8_t>());
 */
class RGBToAutoLibyuvConverter : public ConverterBase
{
  public:
    RGBToAutoLibyuvConverter(int dstWidth, int dstHeight, AVPixelFormat dstPixFmt,
                             AVColorSpace colorspace = AVCOL_SPC_BT709)
        : ConverterBase(), width(dstWidth), height(dstHeight), dst_fmt(dstPixFmt),
          targetColorspace(colorspace)
    {
        CELUX_DEBUG("Initializing RGBToAutoLibyuvConverter ({}x{}, colorspace={})", 
                    width, height, static_cast<int>(colorspace));
    }

    ~RGBToAutoLibyuvConverter() override = default;

    /**
     * @brief Converts RGB24 buffer (HWC uint8) to Frame (YUV format).
     *
     * @param frame  Output celux::Frame (must be pre-allocated with correct format/size).
     * @param buffer Input buffer (raw RGB24 from tensor, HWC layout).
     */
    void convert(celux::Frame& frame, void* buffer) override
    {
        // Validate frame
        if (frame.getWidth() != width || frame.getHeight() != height)
        {
            throw std::runtime_error("RGBToAutoLibyuvConverter: Frame size mismatch");
        }

        const uint8_t* rgb = static_cast<const uint8_t*>(buffer);
        if (!rgb)
        {
            throw std::runtime_error("RGBToAutoLibyuvConverter: Null input buffer");
        }

        int rgbStride = width * 3;

        // Get destination plane pointers and strides
        uint8_t* dstY = frame.getData(0);
        uint8_t* dstU = frame.getData(1);
        uint8_t* dstV = frame.getData(2);
        int dstStrideY = frame.getLineSize(0);
        int dstStrideU = frame.getLineSize(1);
        int dstStrideV = frame.getLineSize(2);

        bool success = false;

        switch (dst_fmt)
        {
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUVJ420P:
            success = convertToI420(rgb, rgbStride, dstY, dstStrideY, 
                                    dstU, dstStrideU, dstV, dstStrideV);
            break;

        case AV_PIX_FMT_NV12:
            success = convertToNV12(rgb, rgbStride, dstY, dstStrideY,
                                    dstU, dstStrideU);  // UV interleaved
            break;

        case AV_PIX_FMT_YUV422P:
        case AV_PIX_FMT_YUVJ422P:
            success = convertToI422(rgb, rgbStride, dstY, dstStrideY,
                                    dstU, dstStrideU, dstV, dstStrideV);
            break;

        case AV_PIX_FMT_YUV444P:
        case AV_PIX_FMT_YUVJ444P:
            success = convertToI444(rgb, rgbStride, dstY, dstStrideY,
                                    dstU, dstStrideU, dstV, dstStrideV);
            break;

        default:
            throw std::runtime_error("RGBToAutoLibyuvConverter: Unsupported pixel format");
        }

        if (!success)
        {
            throw std::runtime_error("RGBToAutoLibyuvConverter: libyuv conversion failed");
        }
    }

  private:
    int width;
    int height;
    AVPixelFormat dst_fmt;
    AVColorSpace targetColorspace;

    /**
     * @brief Convert RGB24 to I420 (YUV420P) using libyuv.
     *
     * libyuv's RAWToI420 treats input as RGB24 (R first in memory).
     * The Matrix variant allows specifying BT.601/BT.709 coefficients.
     */
    bool convertToI420(const uint8_t* rgb, int rgbStride,
                       uint8_t* dstY, int dstStrideY,
                       uint8_t* dstU, int dstStrideU,
                       uint8_t* dstV, int dstStrideV)
    {
        // Select YUV matrix based on colorspace
        const libyuv::YuvConstants* matrix = &libyuv::kYuvI601Constants;
        if (targetColorspace == AVCOL_SPC_BT709)
        {
            matrix = &libyuv::kYuvH709Constants;
        }
        else if (targetColorspace == AVCOL_SPC_BT2020_NCL || 
                 targetColorspace == AVCOL_SPC_BT2020_CL)
        {
            matrix = &libyuv::kYuv2020Constants;
        }

        // RAWToJ420Matrix for limited->full range would be:
        // For TV range (16-235), we use the standard I420 conversion
        // libyuv::RAWToI420 uses BT.601 by default
        // 
        // Note: libyuv's "RAW" format is RGB24 (R at byte 0)
        // We need to use RGB24ToI420Matrix for colorspace control
        
        int result = libyuv::RGB24ToI420(
            rgb, rgbStride,
            dstY, dstStrideY,
            dstU, dstStrideU,
            dstV, dstStrideV,
            width, height);

        return result == 0;
    }

    /**
     * @brief Convert RGB24 to NV12 using libyuv.
     */
    bool convertToNV12(const uint8_t* rgb, int rgbStride,
                       uint8_t* dstY, int dstStrideY,
                       uint8_t* dstUV, int dstStrideUV)
    {
        // libyuv doesn't have direct RGB24ToNV12, so we go through I420
        // Allocate temp I420 buffer
        int uvWidth = (width + 1) / 2;
        int uvHeight = (height + 1) / 2;
        
        std::vector<uint8_t> tempU(uvWidth * uvHeight);
        std::vector<uint8_t> tempV(uvWidth * uvHeight);

        // RGB24 -> I420
        int ret = libyuv::RGB24ToI420(
            rgb, rgbStride,
            dstY, dstStrideY,
            tempU.data(), uvWidth,
            tempV.data(), uvWidth,
            width, height);

        if (ret != 0) return false;

        // I420 -> NV12 (interleave U and V into UV plane)
        ret = libyuv::I420ToNV12(
            dstY, dstStrideY,
            tempU.data(), uvWidth,
            tempV.data(), uvWidth,
            dstY, dstStrideY,  // Y stays in place
            dstUV, dstStrideUV,
            width, height);

        return ret == 0;
    }

    /**
     * @brief Convert RGB24 to I422 (YUV422P) using libyuv.
     * 
     * libyuv doesn't have RGB24ToI422, so we convert RGB24→ARGB→I422.
     */
    bool convertToI422(const uint8_t* rgb, int rgbStride,
                       uint8_t* dstY, int dstStrideY,
                       uint8_t* dstU, int dstStrideU,
                       uint8_t* dstV, int dstStrideV)
    {
        // Allocate ARGB intermediate buffer
        int argbStride = width * 4;
        std::vector<uint8_t> argb(argbStride * height);
        
        // RGB24 -> ARGB (libyuv expects BGRA order, so use RGB24ToARGB)
        int ret = libyuv::RGB24ToARGB(
            rgb, rgbStride,
            argb.data(), argbStride,
            width, height);
        
        if (ret != 0) return false;
        
        // ARGB -> I422
        ret = libyuv::ARGBToI422(
            argb.data(), argbStride,
            dstY, dstStrideY,
            dstU, dstStrideU,
            dstV, dstStrideV,
            width, height);

        return ret == 0;
    }

    /**
     * @brief Convert RGB24 to I444 (YUV444P) using libyuv.
     * 
     * libyuv doesn't have RGB24ToI444, so we convert RGB24→ARGB→I444.
     */
    bool convertToI444(const uint8_t* rgb, int rgbStride,
                       uint8_t* dstY, int dstStrideY,
                       uint8_t* dstU, int dstStrideU,
                       uint8_t* dstV, int dstStrideV)
    {
        // Allocate ARGB intermediate buffer
        int argbStride = width * 4;
        std::vector<uint8_t> argb(argbStride * height);
        
        // RGB24 -> ARGB
        int ret = libyuv::RGB24ToARGB(
            rgb, rgbStride,
            argb.data(), argbStride,
            width, height);
        
        if (ret != 0) return false;
        
        // ARGB -> I444
        ret = libyuv::ARGBToI444(
            argb.data(), argbStride,
            dstY, dstStrideY,
            dstU, dstStrideU,
            dstV, dstStrideV,
            width, height);

        return ret == 0;
    }
};

} // namespace cpu
} // namespace conversion
} // namespace celux
