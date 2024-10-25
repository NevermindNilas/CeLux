// YUV420P10ToRGB48.hpp
#pragma once

#include "CPUConverter.hpp"
#include "Frame.hpp"
#include <cstring> // For memcpy
#include <iostream>
#include <stdexcept>
#include <type_traits>

extern "C"
{
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

namespace celux
{
namespace conversion
{
namespace cpu
{

/**
 * @brief Converter for YUV420P10LE to RGB48LE conversion on CPU.
 *
 * This converter handles 10-bit YUV420 planar (YUV420P10LE) input and converts it to
 * 48-bit RGB (RGB48LE) output using FFmpeg's sws_scale.
 */
class YUV420P10ToRGB48 : public ConverterBase
{
  public:
    /**
     * @brief Constructor that initializes the base class and swsContext to nullptr.
     */
    YUV420P10ToRGB48() : ConverterBase(), swsContext(nullptr)
    {
    }

    /**
     * @brief Destructor that frees the swsContext if it was initialized.
     */
    ~YUV420P10ToRGB48()
    {
        if (swsContext)
        {
            sws_freeContext(swsContext);
            swsContext = nullptr;
        }
    }

    /**
     * @brief Performs YUV420P10LE to RGB48LE conversion.
     *
     * @param frame Reference to the frame to be converted.
     * @param buffer Pointer to the buffer where RGB48LE data will be stored.
     *
     * @throws std::runtime_error if conversion fails or unsupported formats are
     * provided.
     */
    void convert(celux::Frame& frame, void* buffer) override
    {
        try
        {
            CELUX_DEBUG("ATTEMPTING YUV420P10 CONVERSION");

            // Verify the pixel format
            if (frame.getPixelFormat() != AV_PIX_FMT_YUV420P10LE)
            {
                std::cerr << "Format not YUV420P10LE. Format is actually: "
                          << av_get_pix_fmt_name(frame.getPixelFormat()) << std::endl;
                throw std::invalid_argument(
                    "Unsupported pixel format for YUV420P10ToRGB48 converter.");
            }
            CELUX_DEBUG("Starting YUV420P10ToRGB48 conversion.");

            // Initialize the swsContext if not already done
            if (!swsContext)
            {
                swsContext = sws_getContext(frame.getWidth(), frame.getHeight(),
                                            AV_PIX_FMT_YUV420P10LE, // Source format
                                            frame.getWidth(), frame.getHeight(),
                                            AV_PIX_FMT_RGB48LE, // Destination format
                                            SWS_BILINEAR, nullptr, nullptr, nullptr);
                CELUX_DEBUG("CONTEXT GATHERED FOR 10 BIT YUV");
                if (!swsContext)
                {
                    throw std::runtime_error("Failed to initialize swsContext for "
                                             "YUV420P10LE to RGB48LE conversion");
                }

                // Set color space and range explicitly (optional)
                int srcRange = 0; // MPEG (16-235)
                int dstRange = 1; // JPEG (0-255)
                const int* srcMatrix = sws_getCoefficients(SWS_CS_ITU709);
                const int* dstMatrix = sws_getCoefficients(SWS_CS_ITU709);
                CELUX_DEBUG("Setting colorspace details");
                sws_setColorspaceDetails(swsContext, srcMatrix, srcRange, dstMatrix,
                                         dstRange, 0, 1 << 16, 1 << 16);
            }

            // Source data and line sizes
            const uint8_t* srcData[4] = {nullptr};
            int srcLineSize[4] = {0};

            srcData[0] = frame.getData(0); // Y plane
            srcData[1] = frame.getData(1); // U plane
            srcData[2] = frame.getData(2); // V plane

            srcLineSize[0] = frame.getLineSize(0);
            srcLineSize[1] = frame.getLineSize(1);
            srcLineSize[2] = frame.getLineSize(2);

            // Destination data and line sizes for user buffer
            uint8_t* dstData[4] = {nullptr};
            int dstLineSize[4] = {0};

            // Calculate the required buffer size for RGB48LE
            int numBytes = av_image_get_buffer_size(
                AV_PIX_FMT_RGB48LE, frame.getWidth(), frame.getHeight(), 1);
            CELUX_DEBUG("NUM BYTES: {}", numBytes);
            if (numBytes < 0)
            {
                throw std::runtime_error("Could not get buffer size for RGB48LE");
            }

            CELUX_DEBUG("Setting up av_image_fill_arrays with user buffer");

            if (!buffer)
            {
                throw std::invalid_argument("Destination buffer is null");
            }

            // Initialize the destination data pointers and line sizes to point to the
            // user buffer
            int ret = av_image_fill_arrays(
                dstData, dstLineSize, static_cast<uint8_t*>(buffer), AV_PIX_FMT_RGB48LE,
                frame.getWidth(), frame.getHeight(), 16 // Alignment set to 16
            );
            if (ret < 0)
            {
                char errBuf[256];
                av_strerror(ret, errBuf, sizeof(errBuf));
                throw std::runtime_error(std::string("av_image_fill_arrays failed: ") +
                                         errBuf);
            }
            CELUX_DEBUG("av_image_fill_arrays succeeded");

            // Perform the conversion from YUV420P10LE to RGB48LE directly into user
            // buffer
            CELUX_DEBUG("Starting sws_scale");
            int result = sws_scale(swsContext, srcData, srcLineSize, 0,
                                   frame.getHeight(), dstData, dstLineSize);
            CELUX_DEBUG("sws_scale result: {}", result);

            if (result <= 0)
            {
                throw std::runtime_error(
                    "sws_scale failed during YUV420P10LE to RGB48LE conversion");
            }

            CELUX_DEBUG("Conversion successful");
        }
        catch (const std::exception& e)
        {
            CELUX_DEBUG("Error in 10 bit YUV conversion: {}", e.what());
        }
    }


  private:
    struct SwsContext* swsContext;
};

} // namespace cpu
} // namespace conversion
} // namespace celux
