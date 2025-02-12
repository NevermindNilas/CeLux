#pragma once

#include "CPUConverter.hpp"

namespace celux
{
namespace conversion
{
namespace cpu
{

/**
 * @brief Converter for 12-bit YUV444P12LE to 48-bit RGB (RGB48LE) on CPU.
 */
class YUV444P12ToRGB48 : public ConverterBase
{
  public:
    YUV444P12ToRGB48() : ConverterBase(), swsContext(nullptr)
    {
    }
    ~YUV444P12ToRGB48()
    {
        if (swsContext)
        {
            sws_freeContext(swsContext);
            swsContext = nullptr;
        }
    }

    void convert(celux::Frame& frame, void* buffer) override
    {
        if (frame.getPixelFormat() != AV_PIX_FMT_YUV444P12LE)
        {
            throw std::invalid_argument(
                "Unsupported pixel format for YUV444P12ToRGB48 converter");
        }

        if (!swsContext)
        {
            swsContext = sws_getContext(frame.getWidth(), frame.getHeight(),
                                        AV_PIX_FMT_YUV444P12LE, // source
                                        frame.getWidth(), frame.getHeight(),
                                        AV_PIX_FMT_RGB48LE, // destination
                                        SWS_BILINEAR, nullptr, nullptr, nullptr);

            if (!swsContext)
            {
                throw std::runtime_error("Failed to initialize swsContext for "
                                         "YUV444P12->RGB48 conversion");
            }

            int srcRange = 0;
            int dstRange = 1;
            const int* srcMatrix = sws_getCoefficients(SWS_CS_DEFAULT);
            const int* dstMatrix = sws_getCoefficients(SWS_CS_DEFAULT);
            sws_setColorspaceDetails(swsContext, srcMatrix, srcRange, dstMatrix,
                                     dstRange, 0, 1 << 16, 1 << 16);
        }

        const uint8_t* srcData[4] = {frame.getData(0), frame.getData(1),
                                     frame.getData(2), nullptr};
        int srcLineSize[4] = {frame.getLineSize(0), frame.getLineSize(1),
                              frame.getLineSize(2), 0};

        uint8_t* dstData[4] = {nullptr};
        int dstLineSize[4] = {0};

        int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB48LE, frame.getWidth(),
                                                frame.getHeight(), 1);
        if (numBytes < 0)
        {
            throw std::runtime_error("Could not get buffer size for RGB48LE");
        }

        if (!buffer)
        {
            throw std::invalid_argument("Destination buffer is null");
        }

        int ret = av_image_fill_arrays(
            dstData, dstLineSize, static_cast<uint8_t*>(buffer), AV_PIX_FMT_RGB48LE,
            frame.getWidth(), frame.getHeight(), 1 /* alignment */);
        if (ret < 0)
        {
            char errBuf[256];
            av_strerror(ret, errBuf, sizeof(errBuf));
            throw std::runtime_error(std::string("av_image_fill_arrays failed: ") +
                                     errBuf);
        }

        int result = sws_scale(swsContext, srcData, srcLineSize, 0, frame.getHeight(),
                               dstData, dstLineSize);
        if (result <= 0)
        {
            throw std::runtime_error("sws_scale failed for YUV444P12->RGB48");
        }
    }

  private:
    struct SwsContext* swsContext;
};

} // namespace cpu
} // namespace conversion
} // namespace celux
