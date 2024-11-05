// Decoder.hpp
#pragma once

#include "CxException.hpp"
#include <Conversion.hpp>
#include <Frame.hpp>
#include <FilterFactory.hpp>


namespace celux
{

class Decoder
{
  public:
    struct VideoProperties
    {
        std::string codec;
        int width;
        int height;
        double fps;
        double duration;
        int totalFrames;
        AVPixelFormat pixelFormat;
        bool hasAudio;
        int bitDepth;
        double aspectRatio;     // New property
        int audioBitrate;       // New property
        int audioChannels;      // New property
        int audioSampleRate;    // New property
        std::string audioCodec; // New property
        double min_fps;         // New property for minimum fps
        double max_fps;         // New property for maximum fps
    };

    Decoder() = default;
    // Constructor
    Decoder(int numThreads, std::vector<std::shared_ptr<FilterBase>> filters);
    bool seekToNearestKeyframe(double timestamp);
    // Destructor
    virtual ~Decoder();

    // Deleted copy constructor and assignment operator
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;
    /**
     * @brief Adds a filter to the decoder's filter pipeline.
     *
     * @param filter Shared pointer to a Filter instance.
     */
    void addFilter(const std::unique_ptr<FilterBase>& filter);
    // Move constructor and assignment operator
    Decoder(Decoder&&) noexcept;
    Decoder& operator=(Decoder&&) noexcept;


    /**
     * @brief Decode the next frame and store it in the provided buffer.
     *
     * @param buffer Pointer to the buffer where the frame data will be stored.
     * @param frame_timestamp Optional pointer to a double where the frame's timestamp
     * will be stored.
     * @return true if a frame was successfully decoded, false otherwise.
     */
    virtual bool decodeNextFrame(void* buffer, double* frame_timestamp = nullptr);

    virtual bool seek(double timestamp);
    virtual VideoProperties getVideoProperties() const;
    virtual bool isOpen() const;
    virtual void close();
    virtual std::vector<std::string> listSupportedDecoders() const;
    AVCodecContext* getCtx();

    // getter for bit depth
    int getBitDepth() const;

  protected:
    // Initialization method
    void initialize(const std::string& filePath);
    bool isHardwareAccelerated(const AVCodec* codec);
    void setProperties();
    // Virtual methods for customization
    virtual void openFile(const std::string& filePath);
    virtual void initHWAccel(); // Default does nothing
    virtual void findVideoStream();
    virtual void initCodecContext();
    virtual int64_t convertTimestamp(double timestamp) const;
    void populateProperties();
    void setFormatFromBitDepth();
    // Deleters and smart pointers
    struct AVFormatContextDeleter
    {
        void operator()(AVFormatContext* ctx) const
        {
            avformat_close_input(&ctx);
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

    struct AVPacketDeleter
    {
        void operator()(AVPacket* pkt) const
        {
            av_packet_free(&pkt);
        }
    };
    AVFilterGraph* filter_graph_;
    AVFilterContext* buffersrc_ctx_;
    AVFilterContext* buffersink_ctx_;

    std::vector<std::shared_ptr<FilterBase>> filters_;

    /**
     * @brief Initializes the filter graph based on the added filters.
     *
     * @return true if successful.
     * @return false otherwise.
     */
    bool initFilterGraph();
    using AVFormatContextPtr = std::unique_ptr<AVFormatContext, AVFormatContextDeleter>;
    using AVCodecContextPtr = std::unique_ptr<AVCodecContext, AVCodecContextDeleter>;
    using AVBufferRefPtr = std::unique_ptr<AVBufferRef, AVBufferRefDeleter>;
    using AVPacketPtr = std::unique_ptr<AVPacket, AVPacketDeleter>;
    void set_sw_pix_fmt(AVCodecContextPtr& codecCtx, int bitDepth);
    std::string buildFilterArgs(const AVCodecContext* codecCtx,
                                const AVFormatContext* formatCtx, int videoStreamIndex) const;
    std::string ptrToHexString(void* ptr) const;
    // Member variables
    AVFormatContextPtr formatCtx;
    AVCodecContextPtr codecCtx;
    AVPacketPtr pkt;
    int videoStreamIndex;
    VideoProperties properties;
    Frame frame;
    bool isHwAccel;
    std::unique_ptr<celux::conversion::IConverter> converter;
    AVBufferRefPtr hwDeviceCtx; // For hardware acceleration
    AVBufferRefPtr hwFramesCtx; // For hardware acceleration
    int numThreads;
    /**
     * @brief Get the timestamp of the frame in seconds.
     *
     * @param frame Pointer to the AVFrame.
     * @return double Timestamp in seconds.
     */
    double getFrameTimestamp(AVFrame* frame);
};
} // namespace celux
