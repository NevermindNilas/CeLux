#pragma once

#include "error/CxException.hpp"
#include <Conversion.hpp>
#include <Frame.hpp>
#include <torch/torch.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

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
        double aspectRatio;
        int audioBitrate;
        int audioChannels;
        int audioSampleRate;
        std::string audioCodec;
        double min_fps;
        double max_fps;
    };

    Decoder() = default;
    Decoder(int numThreads);
    bool seekToNearestKeyframe(double timestamp);
    virtual ~Decoder();

    // Deleted copy constructor and assignment operator
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    Decoder(Decoder&&) noexcept;
    Decoder& operator=(Decoder&&) noexcept;
    bool seekFrame(int frameIndex);
    virtual bool decodeNextFrame(void* buffer, double* frame_timestamp = nullptr);
    virtual bool seek(double timestamp);
    virtual VideoProperties getVideoProperties() const;
    virtual bool isOpen() const;
    virtual void close();
    void setLibyuvEnabled(bool enabled);
    void setForce8Bit(bool enabled);
    int getBitDepth() const;

    virtual std::vector<std::string> listSupportedDecoders() const;
    AVCodecContext* getCtx();

    bool extractAudioToFile(const std::string& outputFilePath);
    torch::Tensor getAudioTensor();

  protected:
    void initialize(const std::string& filePath);
    void setProperties();
    virtual void openFile(const std::string& filePath);
    virtual void findVideoStream();
    virtual void initCodecContext();
    virtual int64_t convertTimestamp(double timestamp) const;

    double getFrameTimestamp(AVFrame* frame);

    std::unique_ptr<celux::conversion::IConverter> converter;
    std::unique_ptr<AVFormatContext, AVFormatContextDeleter> formatCtx;
    std::unique_ptr<AVCodecContext, AVCodecContextDeleter> codecCtx;
    std::unique_ptr<AVPacket, AVPacketDeleter> pkt;
    int videoStreamIndex;
    int numThreads;
    VideoProperties properties;
    Frame frame;
    bool libyuv_enabled = true;
    bool force_8bit = false;
    int audioStreamIndex = -1;
    AVCodecContextPtr audioCodecCtx;
    Frame audioFrame;
    AVPacketPtr audioPkt;
    SwrContextPtr swrCtx;

    bool initializeAudio();
    void closeAudio();

    std::thread decodingThread;
    std::atomic<bool> stopDecoding{false};
    std::atomic<bool> seekRequested{false};
    std::queue<Frame> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCond;
    std::condition_variable producerCond;
    size_t maxQueueSize = 20;
    bool isFinished = false;

    void decodingLoop();
    void startDecodingThread();
    void stopDecodingThread();
    void clearQueue();
};
} // namespace celux
