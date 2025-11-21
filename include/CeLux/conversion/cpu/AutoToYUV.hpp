#pragma once

extern "C"
{
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

#include "CPUConverter.hpp"
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace celux
{
namespace conversion
{
namespace cpu
{

/**
 * @brief Robust converter dynamically handling pixel formats to YUV420P (8-bit) or YUV420P16LE (10/12-bit).
 *        Output is planar Y, U, V packed into a single contiguous buffer.
 */
class AutoToYUVConverter : public ConverterBase
{
  public:
    AutoToYUVConverter()
        : ConverterBase(), sws_ctx(nullptr), last_src_fmt(AV_PIX_FMT_NONE),
          last_dst_fmt(AV_PIX_FMT_NONE), last_src_colorspace(AVCOL_SPC_UNSPECIFIED),
          last_src_color_range(AVCOL_RANGE_UNSPECIFIED), last_width(0), last_height(0)
    {
    }

    ~AutoToYUVConverter() override
    {
        if (sws_ctx)
            sws_freeContext(sws_ctx);
    }

    void convert(celux::Frame& frame, void* buffer) override
    {
        AVFrame* av_frame = frame.get();
        const AVPixelFormat src_fmt = frame.getPixelFormat();
        const int width = frame.getWidth();
        const int height = frame.getHeight();

        // 1) Derive effective bit depth from the frame itself
        const int bit_depth = effective_bit_depth_from_frame(av_frame);

        // 2) Choose destination format
        // YUV420P for 8-bit, YUV420P16LE for >8-bit
        const AVPixelFormat dst_fmt =
            (bit_depth <= 8) ? AV_PIX_FMT_YUV420P : AV_PIX_FMT_YUV420P16LE;
        
        const int elem_size = (bit_depth <= 8) ? 1 : 2; // bytes per pixel component

        // 3) Colorspace defaults
        AVColorSpace src_colorspace = av_frame->colorspace;
        if (src_colorspace == AVCOL_SPC_UNSPECIFIED)
            src_colorspace = (height > 576) ? AVCOL_SPC_BT709 : AVCOL_SPC_BT470BG;

        AVColorRange src_color_range = av_frame->color_range;
        if (src_color_range == AVCOL_RANGE_UNSPECIFIED)
            src_color_range = AVCOL_RANGE_MPEG;

        // 4) (Re)build sws context if anything changed
        if (!sws_ctx || src_fmt != last_src_fmt || dst_fmt != last_dst_fmt ||
            src_colorspace != last_src_colorspace ||
            src_color_range != last_src_color_range || width != last_width ||
            height != last_height)
        {
            if (sws_ctx)
            {
                sws_freeContext(sws_ctx);
                sws_ctx = nullptr;
            }

            sws_ctx = sws_getContext(width, height, src_fmt, width, height, dst_fmt,
                                     SWS_BICUBIC, nullptr, nullptr, nullptr);
            if (!sws_ctx)
                throw std::runtime_error("Failed to initialize swsContext");

            const int* srcCoeffs = sws_getCoefficients(src_colorspace);
            const int* dstCoeffs = sws_getCoefficients(AVCOL_SPC_BT709); // Normalize to BT.709? Or keep original?
            // The plan mentioned normalizing to a standard color space. BT.709 is a safe bet for HD/4K.
            
            const int srcRange = (src_color_range == AVCOL_RANGE_JPEG) ? 1 : 0;
            // We want to preserve full range if possible, or normalize. 
            // If we output YUV, we usually want standard range (MPEG) unless we explicitly handle full range.
            // However, the "Full Range Hack" discussion suggests we want to preserve values.
            // Let's stick to standard conversion for now, relying on Float32 later to handle range.
            // Actually, sws_scale will handle range conversion if we tell it.
            
            int ok = sws_setColorspaceDetails(sws_ctx, srcCoeffs, srcRange, dstCoeffs,
                                              0, 0, 1 << 16, 1 << 16); // dstRange=0 (MPEG)
            if (ok < 0)
                CELUX_WARN("sws_setColorspaceDetails returned {}", ok);

            last_src_fmt = src_fmt;
            last_dst_fmt = dst_fmt;
            last_src_colorspace = src_colorspace;
            last_src_color_range = src_color_range;
            last_width = width;
            last_height = height;
        }

        // 5) Calculate offsets for Y, U, V planes in the contiguous buffer
        uint8_t* dst_base = static_cast<uint8_t*>(buffer);
        
        // Y plane
        uint8_t* dst_y = dst_base;
        int stride_y = width * elem_size;
        size_t size_y = stride_y * height;

        // U plane (subsampled 2x2)
        uint8_t* dst_u = dst_y + size_y;
        int stride_u = (width / 2) * elem_size;
        size_t size_u = stride_u * (height / 2);

        // V plane (subsampled 2x2)
        uint8_t* dst_v = dst_u + size_u;
        int stride_v = (width / 2) * elem_size;

        uint8_t* dstData[4] = {dst_y, dst_u, dst_v, nullptr};
        int dstLineSize[4] = {stride_y, stride_u, stride_v, 0};

        const uint8_t* srcData[4] = {av_frame->data[0], av_frame->data[1],
                                     av_frame->data[2], av_frame->data[3]};
        const int srcLineSize[4] = {av_frame->linesize[0], av_frame->linesize[1],
                                    av_frame->linesize[2], av_frame->linesize[3]};

        const int result =
            sws_scale(sws_ctx, srcData, srcLineSize, 0, height, dstData, dstLineSize);
        if (result != height)
            throw std::runtime_error("sws_scale failed or incomplete");
    }

  private:
    SwsContext* sws_ctx;
    AVPixelFormat last_src_fmt;
    AVPixelFormat last_dst_fmt;
    AVColorSpace last_src_colorspace;
    AVColorRange last_src_color_range;
    int last_width, last_height;
};

} // namespace cpu
} // namespace conversion
} // namespace celux
