// BGRAToRGB.hpp
#pragma once

#include "CPUConverter.hpp"


namespace celux
{
namespace conversion
{
namespace cpu
{

/**
 * @brief Converter for BGRA to RGB24 conversion on CPU.
 */
class BGRAToRGB : public ConverterBase
{
  public:
    BGRAToRGB() : ConverterBase(), swsContext(nullptr)
    {
        CELUX_DEBUG("Initializing BGRAToRGB converter");
    }

    ~BGRAToRGB()
    {
        if (swsContext)
        {
            sws_freeContext(swsContext);
            swsContext = nullptr;
        }
    }

    void convert(celux::Frame& frame, void* buffer) override
    {
        if (frame.getPixelFormat() != AV_PIX_FMT_BGRA)
        {
            std::cerr << "Format not BGRA. Format is actually: "
                      << av_get_pix_fmt_name(frame.getPixelFormat()) << std::endl;
            throw std::invalid_argument(
                "Unsupported pixel format for BGRAToRGB converter.");
        }

        if (!swsContext)
        {
            swsContext = sws_getContext(frame.getWidth(), frame.getHeight(),
                                        AV_PIX_FMT_BGRA, // Source format
                                        frame.getWidth(), frame.getHeight(),
                                        AV_PIX_FMT_RGB24, // Destination format
                                        SWS_BILINEAR, nullptr, nullptr, nullptr);

            if (!swsContext)
            {
                throw std::runtime_error(
                    "Failed to initialize swsContext for BGRA to RGB24 conversion");
            }

            sws_setColorspaceDetails(swsContext, sws_getCoefficients(SWS_CS_DEFAULT), 0,
                                     sws_getCoefficients(SWS_CS_DEFAULT), 1, 0, 1 << 16,
                                     1 << 16);
        }

        const uint8_t* srcData[4] = {nullptr};
        int srcLineSize[4] = {0};
        srcData[0] = frame.getData(0); // BGRA data
        srcLineSize[0] = frame.getLineSize(0);

        uint8_t* dstData[4] = {nullptr};
        int dstLineSize[4] = {0};

        int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, frame.getWidth(),
                                                frame.getHeight(), 1);

        if (numBytes < 0)
        {
            throw std::runtime_error("Could not get buffer size for RGB24");
        }

        int ret = av_image_fill_arrays(dstData, dstLineSize,
                                       static_cast<uint8_t*>(buffer), AV_PIX_FMT_RGB24,
                                       frame.getWidth(), frame.getHeight(), 1);

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
            throw std::runtime_error(
                "sws_scale failed during BGRA to RGB24 conversion");
        }
    }

  private:
    struct SwsContext* swsContext;
};

} // namespace cpu
} // namespace conversion
} // namespace celux
