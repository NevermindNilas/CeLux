// Python/VideoReader.cpp
#include <torch/extension.h>
#include "python/VideoReader.hpp"
#include <pybind11/pybind11.h>
#include <torch/torch.h> // Ensure you have included the necessary Torch headers

namespace py = pybind11;
#define CHECK_TENSOR(tensor)                                                  \
	if (!tensor.defined() || tensor.numel() == 0)                              \
	{                                                                         \
		throw std::runtime_error("Invalid tensor: undefined or empty");        \
	}

VideoReader::VideoReader(const std::string& filePath, int numThreads, bool force_8bit)
    : decoder(nullptr), rand_decoder(nullptr), currentIndex(0), start_frame(0),
      end_frame(-1), start_time(-1.0), end_time(-1.0), filePath(filePath),
      numThreads(numThreads), force_8bit(force_8bit)
{
    CELUX_INFO("VideoReader constructor called with filePath: {}", filePath);

    if (numThreads > std::thread::hardware_concurrency())
        throw std::invalid_argument(
            "Number of threads cannot exceed hardware concurrency");

    try
    {
        torch::Device torchDevice = torch::Device(torch::kCPU);

        // Main sequential decoder
        decoder = celux::Factory::createDecoder(torchDevice, filePath, numThreads);
        decoder->setForce8Bit(force_8bit);
        CELUX_INFO("Main decoder created successfully");

        // Random-access decoder is now lazy-loaded in ensureRandDecoder()
        // rand_decoder = celux::Factory::createDecoder(torchDevice, filePath, numThreads);
        // CELUX_INFO("Random-access decoder created successfully");

        audio = std::make_shared<Audio>(decoder);

        properties = decoder->getVideoProperties();

        torch::Dtype torchDataType = findTypeFromBitDepth();
        tensor = torch::empty(
            {properties.height, properties.width, 3},
            torch::TensorOptions().dtype(torchDataType).device(torchDevice));
        CHECK_TENSOR(tensor);
    }
    catch (const std::exception& ex)
    {
        CELUX_ERROR("Exception in VideoReader constructor: {}", ex.what());
        throw;
    }
}

void VideoReader::close()
{
    CELUX_INFO("Closing VideoReader");
    if (decoder)
    {
        decoder->close();
        decoder.reset();
    }
    if (rand_decoder)
    {
        rand_decoder->close();
        rand_decoder.reset();
    }
    CELUX_INFO("All decoders closed");
}


std::shared_ptr<VideoReader::Audio> VideoReader::getAudio()
{
    return audio;
}

// -------------------------
// Audio Class Implementation
// -------------------------

VideoReader::Audio::Audio(std::shared_ptr<celux::Decoder> decoder)
    : decoder(std::move(decoder))
{
    if (!this->decoder)
    {
        throw std::runtime_error("Audio: Invalid decoder instance provided.");
    }
}

torch::Tensor VideoReader::Audio::getAudioTensor()
{
    return decoder->getAudioTensor();
}

bool VideoReader::Audio::extractToFile(const std::string& outputFilePath)
{
    return decoder->extractAudioToFile(outputFilePath);
}

celux::Decoder::VideoProperties VideoReader::Audio::getProperties() const
{
    return decoder->getVideoProperties();
}

VideoReader::~VideoReader()
{
    CELUX_INFO("VideoReader destructor called");
    close();
}

void VideoReader::setRange(std::variant<int, double> start,
                           std::variant<int, double> end)
{
    // Check if both start and end are of the same type
    if (start.index() != end.index())
    {

        throw std::invalid_argument("Start and end must be of the same type.");
    }

    // Set the range based on the type of start and end
    if (std::holds_alternative<int>(start) && std::holds_alternative<int>(end))
    {
        int startFrame = std::get<int>(start);
        int endFrame = std::get<int>(end);
        setRangeByFrames(startFrame, endFrame);
    }
    else if (std::holds_alternative<double>(start) &&
             std::holds_alternative<double>(end))
    {
        double startTime = std::get<double>(start);
        double endTime = std::get<double>(end);
        setRangeByTimestamps(startTime, endTime);
    }
    else
    {

        throw std::invalid_argument("Unsupported type for start and end.");
    }
}

void VideoReader::setRangeByFrames(int startFrame, int endFrame)
{
    CELUX_INFO("Setting frame range: start={}, end={}", startFrame, endFrame);

    // Handle negative indices by converting them to positive frame numbers
    if (startFrame < 0)
    {
        startFrame = properties.totalFrames + startFrame;
        CELUX_INFO("Adjusted start_frame to {}", startFrame);
    }
    if (endFrame < 0)
    {
        endFrame = properties.totalFrames + endFrame;
        CELUX_INFO("Adjusted end_frame to {}", endFrame);
    }

    // Validate the adjusted frame range
    if (startFrame < 0 || endFrame < 0)
    {
        CELUX_ERROR("Frame indices out of range after adjustment: start={}, end={}",
                    startFrame, endFrame);
        throw std::runtime_error("Frame indices out of range after adjustment.");
    }
    if (endFrame <= startFrame)
    {
        CELUX_ERROR("Invalid frame range: end_frame ({}) must be greater than "
                    "start_frame ({}) after adjustment.",
                    endFrame, startFrame);
        throw std::runtime_error(
            "end_frame must be greater than start_frame after adjustment.");
    }

    // Make end_frame exclusive by subtracting one
    endFrame = endFrame - 1;
    CELUX_INFO("Adjusted end_frame to be exclusive: {}", endFrame);

    start_frame = startFrame;
    end_frame = endFrame;
    CELUX_INFO("Frame range set: start_frame={}, end_frame={}", start_frame, end_frame);
}

void VideoReader::setRangeByTimestamps(double startTime, double endTime)
{
    CELUX_INFO("Setting timestamp range: start={}, end={}", startTime, endTime);

    // Validate the timestamp range
    if (startTime < 0 || endTime < 0)
    {
        CELUX_ERROR("Timestamps cannot be negative: start={}, end={}", startTime,
                    endTime);
        throw std::invalid_argument("Timestamps cannot be negative.");
    }
    if (endTime <= startTime)
    {
        CELUX_ERROR("Invalid timestamp range: end ({}) must be greater than start ({})",
                    endTime, startTime);
        throw std::invalid_argument("end must be greater than start.");
    }

    // Set the timestamp range
    start_time = startTime;
    end_time = endTime;
    CELUX_INFO("Timestamp range set: start_time={}, end_time={}", start_time, end_time);
}

torch::Tensor VideoReader::readFrame()
{
    CELUX_TRACE("readFrame() called");
    py::gil_scoped_release release; // Release GIL before calling decoder

    double frame_timestamp = 0.0;
    bool success = decoder->decodeNextFrame(tensor.data_ptr(), &frame_timestamp);
    if (!success)
    {
        CELUX_WARN("Decoding failed or no more frames available");
        return torch::Tensor(); // Return an empty tensor if decoding failed
    }

    // Update current timestamp
    current_timestamp = frame_timestamp;

    CELUX_TRACE("Frame decoded successfully at timestamp: {}", current_timestamp);
    return tensor;
}

torch::Tensor VideoReader::makeLikeOutputTensor() const
{
    return torch::empty(
        {properties.height, properties.width, 3},
        torch::TensorOptions().dtype(tensor.dtype()).device(tensor.device()));
}


bool VideoReader::seek(double timestamp)
{
    CELUX_TRACE("Seeking to timestamp: {}", timestamp);

    if (timestamp < 0 || timestamp > properties.duration)
    {
        CELUX_ERROR("Timestamp out of range: {}", timestamp);
        return false;
    }

    bool success = decoder->seekToNearestKeyframe(timestamp);
    if (!success)
    {
        CELUX_WARN("Seek to keyframe failed at timestamp {}", timestamp);
        return false;
    }

    // Decode frames until reaching the exact timestamp
    while (current_timestamp < timestamp)
    {
        readFrame();
    }

    CELUX_TRACE("Exact seek to timestamp {} successful", timestamp);
    return true;
}

std::vector<std::string> VideoReader::supportedCodecs()
{
    CELUX_TRACE("supportedCodecs() called");
    std::vector<std::string> codecs = decoder->listSupportedDecoders();
    CELUX_INFO("Number of supported decoders: {}", codecs.size());
    for (const auto& codec : codecs)
    {
        CELUX_TRACE("Supported decoder: {}", codec);
    }
    return codecs;
}
py::dict VideoReader::getProperties() const
{
    CELUX_TRACE("getProperties() called");
    py::dict props;
    props["width"] = properties.width;
    props["height"] = properties.height;
    props["fps"] = properties.fps;
    props["min_fps"] = properties.min_fps; // New property
    props["max_fps"] = properties.max_fps; // New property
    props["duration"] = properties.duration;
    props["total_frames"] = properties.totalFrames;
    props["pixel_format"] = av_get_pix_fmt_name(properties.pixelFormat)
                                ? av_get_pix_fmt_name(properties.pixelFormat)
                                : "Unknown";
    props["has_audio"] = properties.hasAudio;
    props["audio_bitrate"] = properties.audioBitrate;        // New property
    props["audio_channels"] = properties.audioChannels;      // New property
    props["audio_sample_rate"] = properties.audioSampleRate; // New property
    props["audio_codec"] = properties.audioCodec;            // New property
    props["bit_depth"] = properties.bitDepth;
    props["aspect_ratio"] = properties.aspectRatio; // New property
    props["codec"] = properties.codec;

    CELUX_INFO("Video properties retrieved and converted to Python dict");
    return props;
}

py::object VideoReader::operator[](py::object key)
{
    const double dt = frameDuration();              // seconds per frame
    const double tol = (dt > 0.0) ? dt * 1.1 : 0.0; // ~1 frame tolerance
    const double half = (dt > 0.0) ? 0.5 * dt : 0.0;

    auto clamp_ts = [&](double ts) -> double
    {
        if (ts < 0.0)
            ts = 0.0;
        const double eps = std::max(1e-3, half); // guard near tail
        if (properties.duration > 0.0)
            ts = std::min(ts, std::max(0.0, properties.duration - eps));
        return ts;
    };

    auto norm_idx = [&](long long idx) -> long long
    {
        if (idx < 0)
            idx = static_cast<long long>(properties.totalFrames) +
                  idx; // Pythonic negatives
        return idx;
    };

    auto check_idx_range = [&](long long idx)
    {
        if (idx < 0 || idx >= properties.totalFrames)
        {
            throw py::index_error("Frame index out of range: " + std::to_string(idx));
        }
    };

    auto check_ts_range = [&](double ts)
    {
        if (ts < 0.0 || (properties.duration > 0.0 && ts > properties.duration))
        {
            throw py::value_error("Timestamp out of range: " + std::to_string(ts));
        }
    };

    auto is_tail = [&](double ts) -> bool
    {
        const double eps = std::max(1e-3, half);
        return (properties.duration > 0.0) && (ts >= properties.duration - eps);
    };

    auto advance_until_timestamp = [&](double target_ts) -> torch::Tensor
    {
        const int cap =
            (properties.fps > 0.0) ? static_cast<int>(properties.fps) + 8 : 8;
        torch::Tensor f;
        for (int i = 0; i < cap; ++i)
        {
            f = readFrame();
            if (!f.defined() || f.numel() == 0)
            {
                // Try random-access fallback at a safe ts
                try
                {
                    return frameAt(clamp_ts(target_ts));
                }
                catch (...)
                {
                    return torch::Tensor();
                }
            }
            if (current_timestamp + 1e-9 >= target_ts - half)
                return f;
        }
        // Safety cap hit → try random-access
        try
        {
            return frameAt(clamp_ts(target_ts));
        }
        catch (...)
        {
            return torch::Tensor();
        }
    };

    // ----- int index -----
    if (py::isinstance<py::int_>(key))
    {
        long long req = norm_idx(key.cast<long long>());
        check_idx_range(req); // do NOT silently clamp; be Pythonic

        const double target_ts =
            (properties.fps > 0.0) ? (static_cast<double>(req) / properties.fps) : 0.0;

        const double diff = std::abs(target_ts - current_timestamp);
        if (diff <= tol)
        {
            torch::Tensor f = advance_until_timestamp(target_ts);
            if (!f.defined() || f.numel() == 0)
            {
                // Optional: treat tail as iteration end (not recommended for
                // __getitem__) if (is_tail(target_ts)) throw py::stop_iteration();
                throw std::runtime_error("Failed to decode frame near requested index");
            }
            return py::cast(f);
        }
        return py::cast(frameAt(static_cast<int>(req)));
    }

    // ----- float timestamp -----
    if (py::isinstance<py::float_>(key))
    {
        double ts = key.cast<double>();
        check_ts_range(ts);
        ts = clamp_ts(ts); // keep away from the ambiguous very last timestamp

        const double diff = std::abs(ts - current_timestamp);
        if (diff <= tol)
        {
            torch::Tensor f = advance_until_timestamp(ts);
            if (!f.defined() || f.numel() == 0)
            {
                throw std::runtime_error(
                    "Failed to decode frame near requested timestamp");
            }
            return py::cast(f);
        }
        return py::cast(frameAt(ts));
    }

    throw std::invalid_argument(
        "__getitem__ expects int (frame index) or float (timestamp seconds).");
}

void VideoReader::reset()
{
    CELUX_TRACE("reset() called: Resetting VideoReader state");
    if (decoder)
    {
        decoder->seek(0.0);
    }
    currentIndex = 0;
    current_timestamp = 0.0;
    hasBufferedFrame = false;
    // Reset range if needed, or keep it? Usually reset() implies full reset, 
    // but here we just reset iteration state.
}

void VideoReader::ensureRandDecoder()
{
    if (!rand_decoder)
    {
        CELUX_INFO("Initializing random-access decoder (lazy load)");
        torch::Device torchDevice = torch::Device(torch::kCPU);
        rand_decoder = celux::Factory::createDecoder(torchDevice, filePath, numThreads);
        rand_decoder->setLibyuvEnabled(libyuv_enabled);
        rand_decoder->setForce8Bit(force_8bit);
    }
}

torch::Tensor VideoReader::frameAt(double timestamp_seconds)
{
    ensureRandDecoder();
    if (!rand_decoder)
        throw std::runtime_error("Random-access decoder not initialized");

    // 1. Seek to nearest keyframe before/at target
    {
        py::gil_scoped_release release;
        if (!rand_decoder->seekToNearestKeyframe(timestamp_seconds))
        {
            double backoff = std::max(0.0, timestamp_seconds - 2.0);
            CELUX_WARN("seekToNearestKeyframe({}) failed; retrying with {}",
                       timestamp_seconds, backoff);
            if (!rand_decoder->seekToNearestKeyframe(backoff))
            {
                throw std::runtime_error("Failed to seek in random decoder");
            }
        }
    }

    // 2. Decode forward until we reach the requested timestamp
    torch::Tensor out_frame;
    double hit_ts = -1.0;
    const double half = 0.5 * ((properties.fps > 0) ? 1.0 / properties.fps : 0.0);
    int safety = 0, cap = static_cast<int>(properties.fps * 3) + 16;

    // Allocate buffer once outside the loop
    torch::Tensor buf = makeLikeOutputTensor();

    while (true)
    {
        double ts = 0.0;
        // torch::Tensor buf = makeLikeOutputTensor(); // Moved outside

        bool ok;
        {
            py::gil_scoped_release release;
            ok = rand_decoder->decodeNextFrame(buf.data_ptr(), &ts);
        }
        if (!ok)
            break;

        if (ts + 1e-9 >= timestamp_seconds - half)
        {
            out_frame = buf.clone(); // clone to detach from temp buffer
            hit_ts = ts;
            break;
        }
        if (++safety > cap)
        {
            CELUX_WARN("frameAt(): safety cap hit while advancing to ts={}",
                       timestamp_seconds);
            break;
        }
    }

    if (!out_frame.defined() || out_frame.numel() == 0)
        throw std::runtime_error("Failed to decode frame at requested timestamp");

    CELUX_DEBUG("frameAt(): hit ts={}", hit_ts);
    return out_frame;
}

torch::Tensor VideoReader::frameAt(int frame_index)
{
    CELUX_TRACE("frameAt(index={}) using rand_decoder", frame_index);

    if (frame_index < 0 || frame_index >= properties.totalFrames)
        throw std::out_of_range("Frame index out of range");

    double t = static_cast<double>(frame_index) / std::max(1.0, properties.fps);
    return frameAt(t);
}



bool VideoReader::seekToFrame(int frame_number)
{
    CELUX_INFO("Seeking to frame number: {}", frame_number);

    if (frame_number < 0 || frame_number >= properties.totalFrames)
    {
        CELUX_ERROR("Frame number {} is out of range (0 to {})", frame_number,
                    properties.totalFrames);
        return false;
    }
    double seek_timestamp = frame_number / properties.fps;


    // Seek to the closest keyframe first
    bool success = decoder->seekToNearestKeyframe(seek_timestamp);
    if (!success)
    {
        CELUX_WARN("Seek to keyframe for frame {} failed", frame_number);
        return false;
    }

    // Decode frames until reaching the exact requested frame
    int current_frame = static_cast<int>(current_timestamp * properties.fps);
    while (current_frame < frame_number)
    {
        readFrame();
        current_frame++;
    }

    CELUX_INFO("Exact seek to frame {} successful", frame_number);
    return true;
}


VideoReader& VideoReader::iter()
{
    CELUX_TRACE("iter() called: Preparing VideoReader for iteration");

    // Reset iterator state
    currentIndex = 0;
    current_timestamp = 0.0;
    bufferedFrame = torch::Tensor(); // Clear any old buffered frame
    hasBufferedFrame = false;

    if (start_time >= 0.0 && end_time > 0.0)
    {
        // Using timestamp range
        CELUX_INFO("Using timestamp range for iteration: start_time={}, end_time={}",
                   start_time, end_time);

        bool success = seek(start_time);
        if (!success)
        {
            CELUX_ERROR("Failed to seek to start_time: {}", start_time);
            throw std::runtime_error("Failed to seek to start_time.");
        }

        // -------------------------------------------------------
        // 1) DECODING + DISCARD loop
        // -------------------------------------------------------
        // Keep reading frames, discarding them, until we hit >= start_time.
        while (true)
        {
            // Attempt to decode a frame
            torch::Tensor f = readFrame();
            if (!f.defined() || f.numel() == 0)
            {
                // No more frames, or decode error
                CELUX_WARN("Ran out of frames while discarding up to start_time={}",
                           start_time);
                break;
            }

            // current_timestamp was updated in readFrame().
            if (current_timestamp >= start_time)
            {
                // We have reached or passed start_time
                // --> store this frame for later return in next()
                bufferedFrame = f;
                hasBufferedFrame = true;
                CELUX_DEBUG("Discard loop found first frame at timestamp {}",
                            current_timestamp);
                break;
            }
            // else discard and loop again
        }
        // -------------------------------------------------------

        current_timestamp = std::max(current_timestamp, start_time);
    }
    else if (start_frame >= 0 && end_frame >= 0)
    {
        // Using frame range
        CELUX_INFO("Using frame range for iteration: start_frame={}, end_frame={}",
                   start_frame, end_frame);
        bool success = seekToFrame(start_frame);
        if (!success)
        {
            CELUX_ERROR("Failed to seek to start_frame: {}", start_frame);
            throw std::runtime_error("Failed to seek to start_frame.");
        }
        currentIndex = start_frame;
        current_timestamp = static_cast<double>(currentIndex) / properties.fps;
    }
    else
    {
        // No range set; start from the beginning
        CELUX_INFO("No range set; starting from the beginning");
        bool success = seek(0.0);
        if (!success)
        {
            CELUX_ERROR("Failed to seek to the beginning of the video");
            throw std::runtime_error("Failed to seek to the beginning of the video.");
        }
        current_timestamp = 0.0;
    }

    // Return self for iterator protocol
    return *this;
}

torch::Tensor VideoReader::next()
{
    CELUX_TRACE("next() called: Retrieving next frame");

    // If we have a buffered frame from the discard loop, consume it first.
    torch::Tensor frame;
    if (hasBufferedFrame)
    {
        frame = bufferedFrame;
        hasBufferedFrame = false;
        // current_timestamp is already set by readFrame() earlier.
    }
    else
    {
        // Otherwise decode the next frame
        frame = readFrame();
        if (!frame.defined() || frame.numel() == 0)
        {
            CELUX_INFO("No more frames available (decode returned empty).");
            throw py::stop_iteration();
        }
    }

    // -- Now check if we exceeded the time range AFTER decoding.
    if (start_time >= 0.0 && end_time > 0.0)
    {
        // If the current frame's timestamp is >= end_time, skip/stop. end time + 1 frame
        if (current_timestamp > end_time + 1/properties.fps)
        {
            CELUX_DEBUG("Frame timestamp {} >= end_time {}, skipping frame.",
                        current_timestamp, end_time);
            throw py::stop_iteration();
        }
    }
    else if (start_frame >= 0 && end_frame >= 0)
    {
        if (currentIndex > end_frame)
        {
            CELUX_DEBUG("Frame range exhausted: currentIndex={}, end_frame={}",
                        currentIndex, end_frame);
            throw py::stop_iteration();
        }
    }

    currentIndex++;
    CELUX_TRACE("Returning frame index={}, timestamp={}", currentIndex - 1,
                current_timestamp);
    return frame;
}


void VideoReader::enter()
{
    CELUX_TRACE("enter() called: VideoReader entering context manager");
    // Resources are already initialized in the constructor
    CELUX_INFO("VideoReader is ready for use in context manager");
}

void VideoReader::exit(const py::object& exc_type, const py::object& exc_value,
                       const py::object& traceback)
{
    CELUX_TRACE("exit() called: VideoReader exiting context manager");
    close(); // Close the video reader and free resources
    CELUX_INFO("VideoReader resources have been cleaned up in context manager");
}

int VideoReader::length() const
{
    CELUX_TRACE("length() called: Returning totalFrames = {}", properties.totalFrames);
    return properties.totalFrames;
}

torch::ScalarType VideoReader::findTypeFromBitDepth()
{
    if (force_8bit || libyuv_enabled)
    {
        CELUX_DEBUG("Forcing tensor data type to torch::kUInt8 (force_8bit={} or libyuv_enabled={})", force_8bit, libyuv_enabled);
        return torch::kUInt8;
    }
    int bit_depth = decoder->getBitDepth();
    CELUX_INFO("Bit depth of video: {}", bit_depth);
    torch::ScalarType torchDataType;
    switch (bit_depth)
    {
    case 8:
        CELUX_DEBUG("Setting tensor data type to torch::kUInt8");
        torchDataType = torch::kUInt8;
        break;
    case 10:
        CELUX_DEBUG("Setting tensor data type to torch::kUInt16");
        torchDataType = torch::kUInt16;
        break;
    case 12:
        CELUX_DEBUG("Setting tensor data type to torch::kUInt16");
		torchDataType = torch::kUInt16;
		break;
    case 16:
        CELUX_DEBUG("Setting tensor data type to torch::kUInt16");
        torchDataType = torch::kUInt16;
        break;
    case 32:
        CELUX_DEBUG("Setting tensor data type to torch::kUInt32");
        torchDataType = torch::kUInt32;
        break;
    default:
        CELUX_WARN("Unsupported bit depth: {}", bit_depth);
        throw std::runtime_error("Unsupported bit depth: " + std::to_string(bit_depth));
    }
    return torchDataType;
}


std::shared_ptr<celux::VideoEncoder>
VideoReader::createEncoder(const std::string& outputPath) const
{
    // Build optional audio parameters only if this reader has audio
    std::optional<int> abr = properties.hasAudio
                                 ? std::make_optional(properties.audioBitrate)
                                 : std::nullopt;
    std::optional<int> asr = properties.hasAudio
                                 ? std::make_optional(properties.audioSampleRate)
                                 : std::nullopt;
    std::optional<int> ach = properties.hasAudio
                                 ? std::make_optional(properties.audioChannels)
                                 : std::nullopt;
    std::optional<std::string> acodec =
        properties.hasAudio ? std::make_optional(properties.audioCodec) : std::nullopt;

    // Create and return the matching encoder
    return std::make_shared<celux::VideoEncoder>(
        outputPath,
        /* codec          */ std::nullopt,
        /* width          */ properties.width,
        /* height         */ properties.height,
        /* bitRate        */ std::nullopt,
        /* fps            */ static_cast<float>(properties.fps),
        /* audioBitRate   */ abr,
        /* audioSampleRate*/ asr,
        /* audioChannels  */ ach,
        /* audioCodec     */ acodec);
}

void VideoReader::set_libyuv_enabled(bool enabled)
{
    libyuv_enabled = enabled;
    if (decoder)
        decoder->setLibyuvEnabled(enabled);
    if (rand_decoder)
        rand_decoder->setLibyuvEnabled(enabled);
}

std::string VideoReader::getPixelFormat() const
{
    const char* name = av_get_pix_fmt_name(properties.pixelFormat);
    return name ? std::string(name) : "Unknown";
}
