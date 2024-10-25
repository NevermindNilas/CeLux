#pragma once

#include "CxException.hpp"
#include <Conversion.hpp>
#include <Frame.hpp>

namespace celux
{
class Encoder
{
  public:
    Encoder(const std::string& outputPath, int width, int height, double fps,
            celux::EncodingFormats format, const std::string& codecName,
            std::optional<torch::Stream> stream);

    Encoder() = default;

    virtual ~Encoder();

    // Deleted copy constructor and assignment operator
    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;

    // Move constructor and assignment operator
    Encoder(Encoder&&) noexcept;
    Encoder& operator=(Encoder&&) noexcept;

    // Core methods
    virtual bool encodeFrame(void* buffer);
    virtual bool finalize();
    virtual bool isOpen() const;
    virtual void close();
    virtual std::vector<std::string> listSupportedEncoders() const;
    AVCodecContext* getCtx();

  protected:
    // Initialization method
    void initialize();
    // Virtual methods for customization
    virtual void openFile();
    virtual void initHWAccel(); // Default does nothing
    virtual void initCodecContext(const AVCodec* codec);
    virtual int64_t convertTimestamp(double timestamp) const;
    // Add a new virtual method for configuring codec context
    virtual void configureCodecContext(const AVCodec* codec);

    // Virtual callback for hardware pixel formats
    virtual enum AVPixelFormat getHWFormat(AVCodecContext* ctx,
                                           const enum AVPixelFormat* pix_fmts);

    // Deleters and smart pointers
    struct AVFormatContextDeleter
    {
        void operator()(AVFormatContext* ctx) const
        {
            if (!(ctx->oformat->flags & AVFMT_NOFILE))
                avio_closep(&ctx->pb);
            avformat_free_context(ctx);
        }
    };

    struct AVCodecContextDeleter
    {
        void operator()(AVCodecContext* ctx) const
        {
            avcodec_free_context(&ctx);
        }
    };

    struct AVBufferRefDeleter
    {
        void operator()(AVBufferRef* ref) const
        {
            av_buffer_unref(&ref);
        }
    };

    using AVFormatContextPtr = std::unique_ptr<AVFormatContext, AVFormatContextDeleter>;
    using AVCodecContextPtr = std::unique_ptr<AVCodecContext, AVCodecContextDeleter>;
    using AVBufferRefPtr = std::unique_ptr<AVBufferRef, AVBufferRefDeleter>;

    // Member variables
    AVFormatContextPtr formatCtx;
    AVCodecContextPtr codecCtx;
    AVBufferRefPtr hwDeviceCtx;
    AVBufferRefPtr hwFramesCtx;
    AVStream* stream = nullptr;
    AVPacket* packet = nullptr;
    std::string hwAccelType;
    int64_t pts = 0;
    Frame frame;
    std::unique_ptr<celux::conversion::IConverter> converter;

    //constructor variables
    std::string outputPath;
    int width;
    int height;
    double fps;
    celux::EncodingFormats format;
    std::string codecName;
    std::optional<torch::Stream> encoderStream;
};
} // namespace celux
