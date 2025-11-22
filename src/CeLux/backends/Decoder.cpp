// Decoder.cpp
#include "Decoder.hpp"
#include <Factory.hpp>
#include "conversion/cpu/AutoToRGB.hpp"

using namespace celux::error;

namespace celux
{
Decoder::Decoder(int numThreads)
    : converter(nullptr), formatCtx(nullptr), codecCtx(nullptr), pkt(nullptr),
      videoStreamIndex(-1), numThreads(numThreads)
{
    CELUX_DEBUG("BASE DECODER: Decoder constructed");
}

Decoder::~Decoder()
{
    CELUX_DEBUG("BASE DECODER: Decoder destructor called");
    close();
    closeAudio();
}

Decoder::Decoder(Decoder&& other) noexcept
    : formatCtx(std::move(other.formatCtx)), codecCtx(std::move(other.codecCtx)),
      pkt(std::move(other.pkt)), videoStreamIndex(other.videoStreamIndex),
      properties(std::move(other.properties)), frame(std::move(other.frame)),
      converter(std::move(other.converter))
{
    CELUX_DEBUG("BASE DECODER: Decoder move constructor called");
    other.videoStreamIndex = -1;
    // Reset other members if necessary
}

Decoder& Decoder::operator=(Decoder&& other) noexcept
{
    CELUX_DEBUG("BASE DECODER: Decoder move assignment operator called");
    if (this != &other)
    {
        close();

        formatCtx = std::move(other.formatCtx);
        codecCtx = std::move(other.codecCtx);
        pkt = std::move(other.pkt);
        videoStreamIndex = other.videoStreamIndex;
        properties = std::move(other.properties);
        frame = std::move(other.frame);
        converter = std::move(other.converter);

        other.videoStreamIndex = -1;
        // Reset other members if necessary
    }
    return *this;
}

void Decoder::setProperties()
{
    // Set basic video properties
    properties.codec = codecCtx->codec->name;
    properties.width = codecCtx->width;
    properties.height = codecCtx->height;

    // Frame rate calculation
    properties.fps = av_q2d(formatCtx->streams[videoStreamIndex]->avg_frame_rate);
    properties.min_fps = properties.fps; // Initialize min fps
    properties.max_fps = properties.fps; // Initialize max fps

    // Ensure duration is calculated properly
    if (formatCtx->streams[videoStreamIndex]->duration != AV_NOPTS_VALUE)
    {
        properties.duration =
            static_cast<double>(formatCtx->streams[videoStreamIndex]->duration) *
            av_q2d(formatCtx->streams[videoStreamIndex]->time_base);
    }
    else if (formatCtx->duration != AV_NOPTS_VALUE)
    {
        properties.duration = static_cast<double>(formatCtx->duration) / AV_TIME_BASE;
    }
    else
    {
        properties.duration = 0.0; // Unknown duration
    }

    // Set pixel format and bit depth
    properties.pixelFormat = codecCtx->pix_fmt;
    properties.bitDepth = getBitDepth();

    // Check for audio stream
    properties.hasAudio = false; // Initialize as false
    for (int i = 0; i < formatCtx->nb_streams; ++i)
    {
        if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
        {
            properties.hasAudio = true; // Set to true if an audio stream is found
            properties.audioBitrate = formatCtx->streams[i]->codecpar->bit_rate;
            properties.audioChannels = 
                formatCtx->streams[i]->codecpar->ch_layout.nb_channels;
            properties.audioSampleRate = formatCtx->streams[i]->codecpar->sample_rate;
            properties.audioCodec =
                avcodec_get_name(formatCtx->streams[i]->codecpar->codec_id);
            break; // Stop after finding the first audio stream
        }
    }

    // Calculate total frames
    if (formatCtx->streams[videoStreamIndex]->nb_frames > 0)
    {
        properties.totalFrames = formatCtx->streams[videoStreamIndex]->nb_frames;
    }
    else if (properties.fps > 0 && properties.duration > 0)
    {
        properties.totalFrames = static_cast<int>(properties.fps * properties.duration);
    }
    else
    {
        properties.totalFrames = 0; // Unknown total frames
    }

    // Calculate aspect ratio
    if (properties.width > 0 && properties.height > 0)
    {
        properties.aspectRatio =
            static_cast<double>(properties.width) / properties.height;
    }
    else
    {
        properties.aspectRatio = 0.0; // Unknown aspect ratio
    }

    // Log the video properties
    CELUX_INFO(
        "Video properties: width={}, height={}, fps={}, duration={}, totalFrames={}, "
        "audioBitrate={}, audioChannels={}, audioSampleRate={}, audioCodec={}, "
        "aspectRatio={}",
        properties.width, properties.height, properties.fps, properties.duration,
        properties.totalFrames, properties.audioBitrate, properties.audioChannels,
        properties.audioSampleRate, properties.audioCodec, properties.aspectRatio);
}



void Decoder::initialize(const std::string& filePath)
{
    CELUX_DEBUG("BASE DECODER: Initializing decoder with file: {}", filePath);
    openFile(filePath);
    findVideoStream();
    initCodecContext();
    setProperties();

    converter = std::make_unique<celux::conversion::cpu::AutoToRGBConverter>();
    auto* autoConverter = dynamic_cast<celux::conversion::cpu::AutoToRGBConverter*>(converter.get());
    if (autoConverter)
    {
        autoConverter->setForce8Bit(force_8bit);
        autoConverter->setLibyuvEnabled(libyuv_enabled);
    }

    const AVCodecParameters* params = formatCtx->streams[videoStreamIndex]->codecpar;
    AVColorSpace color_space = params->color_space;         // matrix_coefficients
    AVColorPrimaries colorprim = params->color_primaries;  // color primaries
    AVColorTransferCharacteristic trc = params->color_trc; // transfer curve
    AVColorRange colorrange = params->color_range;         // AVCOL_RANGE_MPEG/JPEG

    CELUX_DEBUG("BASE DECODER: Decoder initialization completed");

    frame.get()->color_range = colorrange;
    frame.get()->color_primaries = colorprim;
    frame.get()->colorspace = color_space;
    frame.get()->color_trc = trc;
    frame.get()->format = params->format;

    CELUX_INFO("BASE DECODER: Decoder using codec: {}, and pixel format: {}",
               codecCtx->codec->name, av_get_pix_fmt_name(codecCtx->pix_fmt));
}

void Decoder::openFile(const std::string& filePath)
{
    CELUX_DEBUG("BASE DECODER: Opening file: {}", filePath);
    // Open input file
    frame = Frame(); // Fallback to CPU Frame

    AVFormatContext* fmt_ctx = nullptr;
    FF_CHECK_MSG(avformat_open_input(&fmt_ctx, filePath.c_str(), nullptr, nullptr),
                 std::string("Failure Opening Input:"));

    formatCtx.reset(fmt_ctx); // Wrap in unique_ptr
    CELUX_DEBUG("BASE DECODER: Input file opened successfully");

    // Retrieve stream information
    FF_CHECK_MSG(avformat_find_stream_info(formatCtx.get(), nullptr),
                 std::string("Failure Finding Stream Info:"));

    pkt.reset(av_packet_alloc()); // Allocate packet
    CELUX_DEBUG("BASE DECODER: Stream information retrieved successfully");
}

void Decoder::findVideoStream()
{
    CELUX_DEBUG("BASE DECODER: Finding best video stream");

    int ret =
        av_find_best_stream(formatCtx.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (ret < 0)
    {
		CELUX_DEBUG("No video stream found");
		throw CxException("No video stream found");
	}

    videoStreamIndex = ret;
    CELUX_DEBUG("BASE DECODER: Video stream found at index {}", videoStreamIndex);
}

void Decoder::initCodecContext()
{
    const AVCodec* codec =
        avcodec_find_decoder(formatCtx->streams[videoStreamIndex]->codecpar->codec_id);
    

    CELUX_DEBUG("BASE DECODER: Initializing codec context");
    if (!codec)
    {
        CELUX_DEBUG("Unsupported codec!");
        throw CxException("Unsupported codec!");
    }

    // Allocate codec context
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx)
    {
        CELUX_DEBUG("Could not allocate codec context");
        throw CxException("Could not allocate codec context");
    }
    codecCtx.reset(codec_ctx);
    CELUX_DEBUG("BASE DECODER: Codec context allocated");


    // Copy codec parameters from input stream to codec context
    FF_CHECK_MSG(avcodec_parameters_to_context(
                     codecCtx.get(), formatCtx->streams[videoStreamIndex]->codecpar),
                 std::string("Failed to copy codec parameters:"));

    CELUX_DEBUG("BASE DECODER: Codec parameters copied to codec context");

    codecCtx->thread_count = numThreads;
    codecCtx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
    CELUX_DEBUG("BASE DECODER: Codec context threading configured: thread_count={}, "
                "thread_type={}",
                codecCtx->thread_count, codecCtx->thread_type);
    codecCtx->time_base = formatCtx->streams[videoStreamIndex]->time_base;
    // Open codec
    FF_CHECK_MSG(avcodec_open2(codecCtx.get(), codec, nullptr),
                 std::string("Failed to open codec:"));

    CELUX_DEBUG("BASE DECODER: Codec opened successfully");
}

// Decoder.cpp

bool Decoder::decodeNextFrame(void* buffer, double* frame_timestamp)
{
    CELUX_TRACE("Decoding next frame");
    int ret;

    if (buffer == nullptr)
    {
        CELUX_DEBUG("Buffer is null");
        throw CxException("Buffer is null");
    }

    while (true)
    {
        // Attempt to receive a decoded frame
        ret = avcodec_receive_frame(codecCtx.get(), frame.get());
        if (ret == AVERROR(EAGAIN))
        {
            CELUX_DEBUG("Decoder needs more packets, reading next packet");
        }
        else if (ret == AVERROR_EOF)
        {
            CELUX_TRACE("No more frames to decode");
            // No more frames to decode
            return false;
        }
        else if (ret < 0)
        {
            CELUX_DEBUG("Error during decoding");
            throw CxException("Error during decoding");
        }
        else
        {
            // Successfully received a frame
            CELUX_DEBUG("Frame decoded successfully");

        

            // **Retrieve the frame timestamp**
            if (frame_timestamp)
            {
                *frame_timestamp = getFrameTimestamp(frame.get());
                CELUX_DEBUG("Frame timestamp retrieved: {}", *frame_timestamp);
            }

            converter->convert(frame, buffer);
            CELUX_DEBUG("Frame converted");

            return true;
        }

        // Read the next packet from the video file
        ret = av_read_frame(formatCtx.get(), pkt.get());
        if (ret == AVERROR_EOF)
        {
            CELUX_TRACE("End of file reached, flushing decoder");
            // End of file: flush the decoder
            FF_CHECK(avcodec_send_packet(codecCtx.get(), nullptr));
        }
        else if (ret < 0)
        {
            CELUX_DEBUG("Error reading frame");
            throw CxException("Error reading frame");
        }
        else
        {
            CELUX_TRACE("Packet read from file, stream_index={}", pkt->stream_index);
            // If the packet belongs to the video stream, send it to the decoder
            if (pkt->stream_index == videoStreamIndex)
            {
                FF_CHECK(avcodec_send_packet(codecCtx.get(), pkt.get()));
                CELUX_DEBUG("Packet sent to decoder");
            }
            else
            {
                CELUX_DEBUG("Packet does not belong to video stream, skipping");
            }
            // Release the packet back to FFmpeg
            av_packet_unref(pkt.get());
        }
    }
}

bool Decoder::seekFrame(int frameIndex)
{
    CELUX_TRACE("Seeking to frame index: {}", frameIndex);

    if (frameIndex < 0 || frameIndex > properties.totalFrames)
    {
        CELUX_WARN("Frame index out of bounds: {}", frameIndex);
        return false;
    }

    int64_t target_pts = av_rescale_q(frameIndex, {1, static_cast<int>(properties.fps)},
                                      formatCtx->streams[videoStreamIndex]->time_base);
    return seek(target_pts * av_q2d(formatCtx->streams[videoStreamIndex]->time_base));
}


bool Decoder::seek(double timestamp)
{
    CELUX_TRACE("Seeking to timestamp: {}", timestamp);
    if (timestamp < 0 || timestamp > properties.duration)
    {
        CELUX_WARN("Timestamp out of bounds: {}", timestamp);
        return false;
    }

    int64_t ts = convertTimestamp(timestamp);
    CELUX_DEBUG("Converted timestamp for seeking: {}", ts);
    int ret = av_seek_frame(formatCtx.get(), videoStreamIndex, ts,
                            AVSEEK_FLAG_BACKWARD);

    if (ret < 0)
    {
        CELUX_DEBUG("Seek failed to timestamp: {}", timestamp);
        return false;
    }

    // Flush codec buffers
    avcodec_flush_buffers(codecCtx.get());
    CELUX_TRACE("Seek successful, codec buffers flushed");

    return true;
}


Decoder::VideoProperties Decoder::getVideoProperties() const
{
    CELUX_TRACE("Retrieving video properties");
    return properties;
}

bool Decoder::isOpen() const
{
    bool open = formatCtx != nullptr && codecCtx != nullptr;
    CELUX_DEBUG("BASE DECODER: Decoder isOpen check: {}", open);
    return open;
}

void Decoder::close()
{
    CELUX_DEBUG("BASE DECODER: Closing decoder");
    if (codecCtx)
    {
        codecCtx.reset();
        CELUX_DEBUG("BASE DECODER: Codec context reset");
    }
    if (formatCtx)
    {
        formatCtx.reset();
        CELUX_DEBUG("BASE DECODER: Format context reset");
    }
    if (converter)
    {
        CELUX_DEBUG("BASE DECODER: Synchronizing converter in Decoder close");
        converter->synchronize();
        converter.reset();
    }
    videoStreamIndex = -1;
    properties = VideoProperties{};
    CELUX_DEBUG("BASE DECODER: Decoder closed");
}

std::vector<std::string> Decoder::listSupportedDecoders() const
{
    CELUX_DEBUG("BASE DECODER: Listing supported decoders");
    std::vector<std::string> decoders;
    void* iter = nullptr;
    const AVCodec* codec = nullptr;

    while ((codec = av_codec_iterate(&iter)) != nullptr)
    {
        if (av_codec_is_decoder(codec))
        {
            std::string codecInfo = std::string(codec->name);

            // Append the long name if available
            if (codec->long_name)
            {
                codecInfo += " - " + std::string(codec->long_name);
            }

            decoders.push_back(codecInfo);
            CELUX_TRACE("Supported decoder found: {}", codecInfo);
        }
    }

    return decoders;
}

AVCodecContext* Decoder::getCtx()
{
    CELUX_TRACE("Getting codec context");
    return codecCtx.get();
}

int64_t Decoder::convertTimestamp(double timestamp) const
{
    CELUX_TRACE("Converting timestamp: {}", timestamp);
    AVRational time_base = formatCtx->streams[videoStreamIndex]->time_base;
    int64_t ts = static_cast<int64_t>(timestamp * time_base.den / time_base.num);
    CELUX_TRACE("Converted timestamp: {}", ts);
    return ts;
}
int Decoder::getBitDepth() const
{
    CELUX_TRACE("Getting bit depth");
    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(
        AVPixelFormat(formatCtx->streams[videoStreamIndex]->codecpar->format));
    if (!desc)
    {
        CELUX_WARN("Unknown pixel format, defaulting to NV12ToRGB");
    }

    int bitDepth = desc->comp[0].depth;
    CELUX_TRACE("Bit depth: {}", bitDepth);
    return bitDepth;
}

bool Decoder::seekToNearestKeyframe(double timestamp)
{
    CELUX_TRACE("Seeking to the nearest keyframe for timestamp: {}", timestamp);
    if (timestamp < 0 || timestamp > properties.duration)
    {
        CELUX_WARN("Timestamp out of bounds: {}", timestamp);
        return false;
    }

    int64_t ts = convertTimestamp(timestamp);
    CELUX_DEBUG("Converted timestamp for keyframe seeking: {}", ts);

    // Perform seek operation to the nearest keyframe before the timestamp
    int ret =
        av_seek_frame(formatCtx.get(), videoStreamIndex, ts, AVSEEK_FLAG_BACKWARD);
    if (ret < 0)
    {
        CELUX_DEBUG("Keyframe seek failed for timestamp: {}", timestamp);
        return false;
    }

    // Flush codec buffers to reset decoding from the keyframe
    avcodec_flush_buffers(codecCtx.get());
    CELUX_TRACE("Keyframe seek successful, codec buffers flushed");

    return true;
}

double Decoder::getFrameTimestamp(AVFrame* frame) 
{
    if (!frame)
    {
        CELUX_WARN("Received a null frame pointer.");
        return -1.0;
    }

    // Define a lambda to convert AV_TIME_BASE to seconds
    auto convert_to_seconds = [&](int64_t timestamp, AVRational time_base) -> double
    { return static_cast<double>(timestamp) * av_q2d(time_base); };

    // Attempt to retrieve the best_effort_timestamp first
    if (frame->best_effort_timestamp != AV_NOPTS_VALUE)
    {
        AVRational time_base = formatCtx->streams[videoStreamIndex]->time_base;
        double timestamp = convert_to_seconds(frame->best_effort_timestamp, time_base);
        CELUX_DEBUG("Using best_effort_timestamp: {}", timestamp);
        return timestamp;
    }

    // Fallback to frame->pts
    if (frame->pts != AV_NOPTS_VALUE)
    {
        AVRational time_base = formatCtx->streams[videoStreamIndex]->time_base;
        double timestamp = convert_to_seconds(frame->pts, time_base);
        CELUX_DEBUG("Using frame->pts: {}", timestamp);
        return timestamp;
    }

    // Fallback to frame->pkt_dts if available
    if (frame->pkt_dts != AV_NOPTS_VALUE)
    {
        AVRational time_base = formatCtx->streams[videoStreamIndex]->time_base;
        double timestamp = convert_to_seconds(frame->pkt_dts, time_base);
        CELUX_DEBUG("Using frame->pkt_dts: {}", timestamp);
        return timestamp;
    }

    // If all timestamp fields are invalid, log a warning and handle accordingly
    CELUX_WARN("Frame has no valid timestamp. Returning -1.0");
    return -1.0;
}
bool Decoder::initializeAudio()
{
    try
    {
        if (!properties.hasAudio)
        {
            CELUX_DEBUG("No audio stream available to initialize.");
            return false;
        }

        CELUX_DEBUG("Initializing audio decoding.");

        // Find the audio stream index if not already set
        if (audioStreamIndex == -1)
        {
            CELUX_DEBUG("Finding audio stream index.");
            for (unsigned int i = 0; i < formatCtx->nb_streams; ++i)
            {
                if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
                {
                    audioStreamIndex = i;
                    CELUX_DEBUG("Audio stream found at index: {}", audioStreamIndex);
                    break;
                }
            }

            if (audioStreamIndex == -1)
            {
                CELUX_DEBUG("Audio stream not found.");
                return false;
            }
        }
        CELUX_DEBUG("Audio stream index: {}", audioStreamIndex);

        AVCodecParameters* codecPar = formatCtx->streams[audioStreamIndex]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(codecPar->codec_id);
        if (!codec)
        {
            CELUX_DEBUG("Unsupported audio codec!");
            return false;
        }

        // Allocate audio codec context
        AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
        if (!codec_ctx)
        {
            CELUX_DEBUG("Could not allocate audio codec context.");
            return false;
        }
        audioCodecCtx.reset(codec_ctx); // Assign to smart pointer

        // Copy codec parameters from input stream to codec context
        if (avcodec_parameters_to_context(audioCodecCtx.get(), codecPar) < 0)
        {
            CELUX_DEBUG("Failed to copy audio codec parameters to context.");
            return false;
        }

        // Open audio codec
        if (avcodec_open2(audioCodecCtx.get(), codec, nullptr) < 0)
        {
            CELUX_DEBUG("Failed to open audio codec.");
            return false;
        }

        // Initialize channel layouts
        AVChannelLayout in_channel_layout = audioCodecCtx->ch_layout;
        if (in_channel_layout.nb_channels == 0)
        {
            // If channel layout is not set, infer from channels
            av_channel_layout_default(&in_channel_layout,
                                      audioCodecCtx->ch_layout.nb_channels);
        }

        AVChannelLayout out_channel_layout;
        av_channel_layout_default(&out_channel_layout, 2); // Stereo output

        AVSampleFormat out_sample_fmt =
            AV_SAMPLE_FMT_S16; // Desired output sample format
        int out_sample_rate = audioCodecCtx->sample_rate; // Desired output sample rate

        // Allocate and set up SwrContext
        SwrContext* swr = nullptr;

        int ret = swr_alloc_set_opts2(&swr,                // Pointer to SwrContext
                                      &out_channel_layout, // Output channel layout
                                      out_sample_fmt,      // Output sample format
                                      out_sample_rate,     // Output sample rate
                                      &in_channel_layout,  // Input channel layout
                                      audioCodecCtx->sample_fmt,  // Input sample format
                                      audioCodecCtx->sample_rate, // Input sample rate
                                      0,                          // Log offset
                                      nullptr                     // Log context
        );

        if (ret < 0 || swr_init(swr) < 0)
        {
            CELUX_DEBUG("Failed to allocate and set SwrContext options: {}",
                        celux::errorToString(ret));
            swr_free(&swr);
            return false;
        }

        swrCtx.reset(swr); // Assign to smart pointer
        CELUX_DEBUG("SwrContext options set successfully.");

        // Allocate audio frame if not already allocated
        if (!audioFrame)
        {
            AVFrame* frame = av_frame_alloc();
            if (!frame)
            {
                CELUX_DEBUG("Could not allocate audio frame.");
                return false;
            }
            audioFrame = Frame(frame);
        }

        // Allocate audio packet if not already allocated
        if (!audioPkt)
        {
            AVPacket* pkt = av_packet_alloc();
            if (!pkt)
            {
                CELUX_DEBUG("Could not allocate audio packet.");
                return false;
            }
            audioPkt.reset(pkt);
        }

        CELUX_DEBUG("Audio decoding initialized successfully.");
        return true;
    }
    catch (const std::exception& e)
    {
        CELUX_DEBUG("Exception occurred during audio initialization: {}", e.what());
        return false;
    }
}

void Decoder::closeAudio()
{
    audioStreamIndex = -1;
    CELUX_DEBUG("Audio decoding resources have been released.");
}


bool Decoder::extractAudioToFile(const std::string& outputFilePath)
{
    CELUX_DEBUG("Starting audio extraction to file: {}", outputFilePath);

    if (!properties.hasAudio)
    {
        CELUX_DEBUG("No audio stream available to extract.");
        return false;
    }

    if (!initializeAudio())
    {
        CELUX_DEBUG("Failed to initialize audio decoding.");
        return false;
    }

    // Reset decoding process
    if (av_seek_frame(formatCtx.get(), audioStreamIndex, 0, AVSEEK_FLAG_BACKWARD) < 0)
    {
        CELUX_DEBUG("Failed to seek audio stream to beginning.");
        return false;
    }
    avcodec_flush_buffers(audioCodecCtx.get());

    // Detect output format
    const AVOutputFormat* outputFormat =
        av_guess_format(nullptr, outputFilePath.c_str(), nullptr);
    if (!outputFormat)
    {
        CELUX_DEBUG("Could not determine output format for: {}", outputFilePath);
        return false;
    }

    // Create output format context
    AVFormatContext* outFormatCtx = nullptr;
    if (avformat_alloc_output_context2(&outFormatCtx, outputFormat, nullptr,
                                       outputFilePath.c_str()) < 0)
    {
        CELUX_DEBUG("Could not allocate output format context.");
        return false;
    }

    // Find encoder for the format
    const AVCodec* audioEncoder = nullptr;
    if (outputFormat->audio_codec == AV_CODEC_ID_MP3)
    {
        audioEncoder = avcodec_find_encoder_by_name("libmp3lame"); // Use LAME for MP3
    }
    else
    {
        audioEncoder = avcodec_find_encoder(outputFormat->audio_codec);
    }

    if (!audioEncoder)
    {
        CELUX_DEBUG("Could not find encoder for format.");
        avformat_free_context(outFormatCtx);
        return false;
    }

    // Create new codec context
    AVCodecContext* audioEncCtx = avcodec_alloc_context3(audioEncoder);
    if (!audioEncCtx)
    {
        CELUX_DEBUG("Could not allocate encoder context.");
        avformat_free_context(outFormatCtx);
        return false;
    }

    // Set codec parameters
    audioEncCtx->bit_rate = 128000;
    audioEncCtx->sample_rate = audioCodecCtx->sample_rate;
    av_channel_layout_default(&audioEncCtx->ch_layout, 2); // Stereo

    // Ensure proper sample format for AAC
    if (outputFormat->audio_codec == AV_CODEC_ID_AAC)
    {
        audioEncCtx->sample_fmt = AV_SAMPLE_FMT_FLTP; // AAC expects float planar
    }
    else
    {
        audioEncCtx->sample_fmt = audioEncoder->sample_fmts
                                      ? audioEncoder->sample_fmts[0]
                                      : AV_SAMPLE_FMT_FLTP;
    }

    audioEncCtx->time_base = {1, audioEncCtx->sample_rate};

    // Open encoder
    if (avcodec_open2(audioEncCtx, audioEncoder, nullptr) < 0)
    {
        CELUX_DEBUG("Could not open encoder.");
        avcodec_free_context(&audioEncCtx);
        avformat_free_context(outFormatCtx);
        return false;
    }

    // Create a new audio stream
    AVStream* audioStream = avformat_new_stream(outFormatCtx, audioEncoder);
    if (!audioStream)
    {
        CELUX_DEBUG("Failed to create audio stream.");
        avcodec_free_context(&audioEncCtx);
        avformat_free_context(outFormatCtx);
        return false;
    }

    audioStream->time_base = {1, audioEncCtx->sample_rate};

    // Copy codec parameters to the stream
    if (avcodec_parameters_from_context(audioStream->codecpar, audioEncCtx) < 0)
    {
        CELUX_DEBUG("Failed to copy encoder parameters.");
        avcodec_free_context(&audioEncCtx);
        avformat_free_context(outFormatCtx);
        return false;
    }

    // Open output file
    if (!(outFormatCtx->oformat->flags & AVFMT_NOFILE))
    {
        if (avio_open(&outFormatCtx->pb, outputFilePath.c_str(), AVIO_FLAG_WRITE) < 0)
        {
            CELUX_DEBUG("Could not open output file: {}", outputFilePath);
            avcodec_free_context(&audioEncCtx);
            avformat_free_context(outFormatCtx);
            return false;
        }
    }

    // Write file header
    if (avformat_write_header(outFormatCtx, nullptr) < 0)
    {
        CELUX_DEBUG("Could not write file header.");
        avcodec_free_context(&audioEncCtx);
        avformat_free_context(outFormatCtx);
        return false;
    }

    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();

    // Read and decode audio
    while (av_read_frame(formatCtx.get(), audioPkt.get()) >= 0)
    {
        if (audioPkt->stream_index != audioStreamIndex)
        {
            av_packet_unref(audioPkt.get());
            continue;
        }

        if (avcodec_send_packet(audioCodecCtx.get(), audioPkt.get()) < 0)
        {
            CELUX_DEBUG("Error sending audio packet.");
            break;
        }

        av_packet_unref(audioPkt.get());

        while (avcodec_receive_frame(audioCodecCtx.get(), frame) >= 0)
        {
            // Check for NaN/Inf values
            for (int ch = 0; ch < frame->ch_layout.nb_channels; ++ch)
            {
                for (int i = 0; i < frame->nb_samples; ++i)
                {
                    float* sample = (float*)frame->data[ch];
                    if (std::isnan(sample[i]) || std::isinf(sample[i]))
                    {
                        sample[i] = 0.0f; // Replace invalid values with silence
                    }
                }
            }

            // Encode the frame
            if (avcodec_send_frame(audioEncCtx, frame) < 0)
            {
                CELUX_DEBUG("Error sending frame to encoder.");
                break;
            }

            while (avcodec_receive_packet(audioEncCtx, pkt) >= 0)
            {
                pkt->stream_index = audioStream->index;
                av_interleaved_write_frame(outFormatCtx, pkt);
                av_packet_unref(pkt);
            }
        }
    }

    // Flush encoder
    avcodec_send_frame(audioEncCtx, nullptr);
    while (avcodec_receive_packet(audioEncCtx, pkt) >= 0)
    {
        pkt->stream_index = audioStream->index;
        av_interleaved_write_frame(outFormatCtx, pkt);
        av_packet_unref(pkt);
    }

    // Write trailer and cleanup
    av_write_trailer(outFormatCtx);
    avcodec_free_context(&audioEncCtx);
    avformat_free_context(outFormatCtx);
    av_frame_free(&frame);
    av_packet_free(&pkt);

    CELUX_DEBUG("Audio extraction completed successfully.");
    return true;
}

torch::Tensor Decoder::getAudioTensor()
{
    CELUX_DEBUG("Starting extraction of audio to torch::Tensor.");

    if (!properties.hasAudio)
    {
        CELUX_DEBUG("No audio stream available to extract.");
        return torch::Tensor();
    }

    if (!initializeAudio())
    {
        CELUX_DEBUG("Failed to initialize audio decoding.");
        return torch::Tensor();
    }

    std::vector<int16_t> audioBuffer; // Assuming S16 format
    // Reset the decoding process before extracting audio
    if (av_seek_frame(formatCtx.get(), audioStreamIndex, 0, AVSEEK_FLAG_BACKWARD) < 0)
    {
        CELUX_DEBUG("Failed to seek audio stream to beginning.");
        return torch::Tensor();
    }
    avcodec_flush_buffers(audioCodecCtx.get());

    // Read and decode audio packets
    while (av_read_frame(formatCtx.get(), audioPkt.get()) >= 0)
    {
        if (audioPkt->stream_index != audioStreamIndex)
        {
            av_packet_unref(audioPkt.get());
            continue;
        }

        // Send packet to decoder
        if (avcodec_send_packet(audioCodecCtx.get(), audioPkt.get()) < 0)
        {
            CELUX_DEBUG("Failed to send audio packet for decoding.");

            return torch::Tensor();
        }

        av_packet_unref(audioPkt.get());

        // Receive all available frames
        while (avcodec_receive_frame(audioCodecCtx.get(), audioFrame.get()) >= 0)
        {
            // Allocate buffer for converted samples
            int dstNbSamples = swr_get_delay(swrCtx.get(), audioCodecCtx->sample_rate) +
                               audioFrame.get()->nb_samples;

            // Allocate buffer for converted samples
            int bufferSize = av_samples_get_buffer_size(
                nullptr, audioCodecCtx->ch_layout.nb_channels,
                audioFrame.get()->nb_samples,
                AV_SAMPLE_FMT_S16, 1);
            if (bufferSize < 0)
            {
                CELUX_DEBUG("Failed to calculate buffer size for audio samples.");

                return torch::Tensor();
            }

            std::vector<uint8_t> buffer(bufferSize);
            uint8_t* out_buffers[] = {buffer.data()};
            // Convert samples to S16
            int convertedSamples =
                swr_convert(swrCtx.get(), out_buffers, audioFrame.get()->nb_samples,
                (const uint8_t**)audioFrame.get()->data, audioFrame.get()->nb_samples);

            if (convertedSamples < 0)
            {
                CELUX_DEBUG("Failed to convert audio samples.");
    
                return torch::Tensor();
            }

            // Append samples to audioBuffer
            int16_t* samples = reinterpret_cast<int16_t*>(buffer.data());
            int numSamples = convertedSamples * audioCodecCtx->ch_layout.nb_channels;
            audioBuffer.insert(audioBuffer.end(), samples, samples + numSamples);
        }
    }

    // Flush decoder
    avcodec_send_packet(audioCodecCtx.get(), nullptr);
    while (avcodec_receive_frame(audioCodecCtx.get(), audioFrame.get()) >= 0)
    {
        // Allocate buffer for converted samples
        int dstNbSamples = swr_get_delay(swrCtx.get(), audioCodecCtx->sample_rate) +
                           audioFrame.get()->nb_samples;

        // Allocate buffer for converted samples
        int bufferSize =
            av_samples_get_buffer_size(nullptr, audioCodecCtx->ch_layout.nb_channels,
                                                    audioFrame.get()->nb_samples,
                                                    AV_SAMPLE_FMT_S16, 1);
        if (bufferSize < 0)
        {
            CELUX_DEBUG(
                "Failed to calculate buffer size for audio samples during flush.");
            return torch::Tensor();
        }

        std::vector<uint8_t> buffer(bufferSize);
        uint8_t* out_buffers[] = {buffer.data()};
        // Convert samples to S16
        int convertedSamples =
            swr_convert(swrCtx.get(), out_buffers, audioFrame.get()->nb_samples,
            (const uint8_t**)audioFrame.get()->data, audioFrame.get()->nb_samples);

        if (convertedSamples < 0)
        {
            CELUX_DEBUG("Failed to convert audio samples during flush.");
            return torch::Tensor();
        }

        // Append samples to audioBuffer
        int16_t* samples = reinterpret_cast<int16_t*>(buffer.data());
        int numSamples = convertedSamples * audioCodecCtx->ch_layout.nb_channels;
        audioBuffer.insert(audioBuffer.end(), samples, samples + numSamples);
    }

    if (audioBuffer.empty())
    {
        CELUX_DEBUG("No audio samples were extracted.");
        return torch::Tensor();
    }

    // Create a Torch tensor from the buffer
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt16);
    torch::Tensor audioTensor =
        torch::from_blob(audioBuffer.data(), {static_cast<long>(audioBuffer.size())},
                         options)
            .clone(); // Clone to ensure the tensor owns its memory

    CELUX_DEBUG("Audio extraction to tensor completed successfully.");
    return audioTensor;
}

void Decoder::setLibyuvEnabled(bool enabled)
{
    libyuv_enabled = enabled;
    if (converter)
    {
        auto* autoConverter = dynamic_cast<celux::conversion::cpu::AutoToRGBConverter*>(converter.get());
        if (autoConverter)
        {
            autoConverter->setLibyuvEnabled(enabled);
        }
    }
}

void Decoder::setForce8Bit(bool enabled)
{
    force_8bit = enabled;
    if (converter)
    {
        auto* autoConverter = dynamic_cast<celux::conversion::cpu::AutoToRGBConverter*>(converter.get());
        if (autoConverter)
        {
            autoConverter->setForce8Bit(enabled);
        }
    }
}

} // namespace celux
