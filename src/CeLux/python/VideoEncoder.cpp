#include "python/VideoEncoder.hpp"
#include <filesystem>
#include <stdexcept>
#include <Factory.hpp>

namespace fs = std::filesystem;

namespace celux
{
    //NOTE --- USED HWC
VideoEncoder::VideoEncoder(const std::string& filename,
                           std::optional<std::string> codec, std::optional<int> width,
                           std::optional<int> height, std::optional<int> bitRate,
                           std::optional<float> fps, std::optional<int> audioBitRate,
                           std::optional<int> audioSampleRate,
                           std::optional<int> audioChannels,
                           std::optional<std::string> audioCodec)
{
    auto properties = inferEncodingProperties(filename, codec, width, height, bitRate,
                                              fps, audioBitRate, audioSampleRate,
                                              audioChannels, audioCodec);
    this->width = properties.width;
    this->height = properties.height;

    encoder = std::make_unique<celux::Encoder>(filename, properties);
}
celux::Encoder::EncodingProperties VideoEncoder::inferEncodingProperties(
    const std::string& filename, std::optional<std::string> codec,
    std::optional<int> width, std::optional<int> height, std::optional<int> bitRate,
    std::optional<float> fps, std::optional<int> audioBitRate,
    std::optional<int> audioSampleRate, std::optional<int> audioChannels,
    std::optional<std::string> audioCodec)
{
    // Populate video encoding settings
    celux::Encoder::EncodingProperties props;
    props.codec = codec.value_or("h264_mf");
    props.width = width.value_or(1920);
    props.height = height.value_or(1080);
    props.bitRate = bitRate.value_or(4000000); // 4 Mbps default
    // fps in VideoEncoder::EncodingProperties is int, so round
    props.fps = static_cast<int>(std::round(fps.value_or(30.0f)));
    props.gopSize = 60;
    props.maxBFrames = 2;
    props.pixelFormat = AV_PIX_FMT_YUV420P;

    // Populate audio encoding settings (0 → no audio)
    if (audioBitRate.has_value() && audioSampleRate.has_value() &&
        audioChannels.has_value() && audioCodec.has_value())
    {
        props.audioBitRate = *audioBitRate;
        props.audioSampleRate = *audioSampleRate;
        props.audioChannels = *audioChannels;
        props.audioCodec = *audioCodec;
    }
    else
    {
        // leave at default-zero if no audio requested
        props.audioBitRate = 0;
        props.audioSampleRate = 0;
        props.audioChannels = 0;
        props.audioCodec = std::string();
    }

    return props;
}

void VideoEncoder::encodeFrame(torch::Tensor frame)
{
    if (!encoder)
        throw std::runtime_error("Encoder is not initialized");

    py::gil_scoped_release release; // <<< release GIL
    celux::Frame convertedFrame;
    convertedFrame.get()->format = AV_PIX_FMT_YUV420P;
    convertedFrame.get()->width = width;
    convertedFrame.get()->height = height;
    convertedFrame.allocateBuffer(32);

    // ✅ Actually move tensor to CPU and make contiguous
    if (frame.device().is_cuda())
    {
        frame = frame.to(torch::kCPU);
    }

    if (!frame.is_contiguous())
    {
        frame = frame.contiguous();
    }


    if (!converter)
    { 
        // Use BT.709 for both source and destination to match AutoToRGB behavior
        // Use Bicubic scaling for better quality
        converter = std::make_unique<celux::conversion::cpu::RGBToAutoConverter>(
            width, height, AV_PIX_FMT_YUV420P,
            1,               // srcRange: Full (RGB)
            0,               // dstRange: Limited (YUV)
            SWS_CS_ITU709,   // srcMatrix: BT.709 (Matches AutoToRGB)
            SWS_CS_ITU709,   // dstMatrix: BT.709 (Matches AutoToRGB)
            SWS_BICUBIC      // scalingFlags: High quality
        );
    }

    // ✅ Pass raw pointer from safe CPU tensor
    converter->convert(convertedFrame, frame.data_ptr<uint8_t>());

    // ✅ Send converted AVFrame to encoder
    encoder->encodeFrame(convertedFrame);
}

void VideoEncoder::close()
{
    if (encoder)
    {
        encoder->close();
        encoder.reset();
    }
}

VideoEncoder::~VideoEncoder()
{
}

// video side stays the same…

/// Replace your old single‑frame binding
void VideoEncoder::encodeAudioFrame(const torch::Tensor& pcm)
{
    py::gil_scoped_release release;

    // 1) grab properties
    auto& props = encoder->Properties();
    int channels = props.audioChannels;
    int sampleRate = props.audioSampleRate;
    int frameSz = encoder->audioFrameSize();
    if (!frameSz)
        throw std::runtime_error("audioFrameSize not set");

    // 2) move to CPU, int16, contiguous
    // move to CPU and cast to Int16 in one go:
    auto t = pcm.to(torch::Device(torch::kCPU), // <— device
                    torch::kInt16,              // <— dtype
                    /*non_blocking=*/false,     // optional
                    /*copy=*/false)             // optional
                 .contiguous();

    auto ptr = t.data_ptr<int16_t>();
    int64_t totalSamples = t.numel() / channels;
    int64_t offset = 0;

    // 3) loop over full‑sized (and final smaller) frames
    while (offset < totalSamples)
    {
        int thisCount = std::min<int64_t>(frameSz, totalSamples - offset);

        // build an AVFrame
        celux::Frame af;
        AVFrame* f = af.get();
        f->nb_samples = thisCount;
        f->sample_rate = sampleRate;
        f->format = AV_SAMPLE_FMT_FLTP; // planar float for AAC
        av_channel_layout_default(&f->ch_layout, channels);
        af.allocateBuffer(0);

        // de‑interleave & convert
        std::vector<float> buf(channels * thisCount);
        for (int ch = 0; ch < channels; ++ch)
        {
            float* dst = buf.data() + ch * thisCount;
            int16_t* src = ptr + (offset * channels) + ch;
            for (int i = 0; i < thisCount; ++i)
                dst[i] = src[i * channels] / 32768.0f;
            std::memcpy(f->data[ch], dst, thisCount * sizeof(float));
        }

        // call the low‑level API
        if (!encoder->encodeAudioFrame(af))
            throw std::runtime_error("audio encode failed");

        offset += thisCount;
    }
}


} // namespace celux
