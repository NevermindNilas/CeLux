// FFCore.hpp
#ifndef CX_CORE_HPP
#define CX_CORE_HPP

#include <algorithm>
#include <cstdint> // For fixed-width integer types
#include <exception>
#include <iostream>
#include <memory>
#include <ostream>   // For std::ostream
#include <stdexcept> // For std::runtime_error
#include <string>
#include <thread>
#include <Logger.hpp>
#include <torch/extension.h>
#include <vector>
#include <optional>
#include <type_traits>
#include <sstream>
#include "json.hpp" // Include the nlohmann/json header
#include <fstream>   // For file I/O
#include <iomanip>   // For std::setprecision
#include <unordered_map>
#include <tuple>
#include <functional>


using json = nlohmann::json;

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h> // For handling channel layout information
#include <libavutil/dict.h>
#include <libavutil/error.h>          // For error codes
#include <libavutil/imgutils.h>       // For image utilities
#include <libavutil/opt.h>            // For AVOptions
#include <libavutil/pixfmt.h>         // For pixel formats
#include <libavutil/samplefmt.h>      // For handling sample format information
#include <libswresample/swresample.h> // Include for SwrContext and resampling functions
                                      // hwaccel

#include <libavutil/hwcontext.h>
    // audio headers
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/audio_fifo.h>
#include <libswscale/swscale.h>

}

// #include <nlohmann/json.hpp>
// Helper function to convert AVOption type to string
static std::string option_type_to_string(int type)
{
    CELUX_INFO("Starting option_type_to_string with type: {}", type);
    switch (type)
    {
    case AV_OPT_TYPE_FLAGS:
        return "Flags";
    case AV_OPT_TYPE_INT:
        return "Integer";
    case AV_OPT_TYPE_INT64:
        return "Integer64";
    case AV_OPT_TYPE_DOUBLE:
        return "Double";
    case AV_OPT_TYPE_FLOAT:
        return "Float";
    case AV_OPT_TYPE_STRING:
        return "String";
    case AV_OPT_TYPE_RATIONAL:
        return "Rational";
    case AV_OPT_TYPE_BINARY:
        return "Binary";
    case AV_OPT_TYPE_DICT:
        return "Dictionary";
    case AV_OPT_TYPE_UINT64:
        return "Unsigned Integer64";
    case AV_OPT_TYPE_CONST:
        return "Constant";
    case AV_OPT_TYPE_IMAGE_SIZE:
        return "Image Size";
    case AV_OPT_TYPE_PIXEL_FMT:
        return "Pixel Format";
    case AV_OPT_TYPE_SAMPLE_FMT:
        return "Sample Format";
    case AV_OPT_TYPE_VIDEO_RATE:
        return "Video Rate";
    case AV_OPT_TYPE_DURATION:
        return "Duration";
    case AV_OPT_TYPE_COLOR:
        return "Color";
    case AV_OPT_TYPE_BOOL:
        return "Boolean";
    case AV_OPT_TYPE_CHLAYOUT:
        return "Channel Layout";
    case AV_OPT_TYPE_FLAG_ARRAY:
        return "Flag Array";
    default:
        CELUX_WARN("Unknown AVOption type: {}", type);
        return "Unknown";
    }
}

// Helper function to convert AVOption flags to string
static std::string option_flags_to_string(int flags)
{
    CELUX_INFO("Starting option_flags_to_string with flags: {}", flags);
    std::string result;
    if (flags & AV_OPT_FLAG_VIDEO_PARAM)
        result += "Video ";
    if (flags & AV_OPT_FLAG_AUDIO_PARAM)
        result += "Audio ";
    if (flags & AV_OPT_FLAG_FILTERING_PARAM)
        result += "Filtering ";
    if (flags & AV_OPT_FLAG_ENCODING_PARAM)
        result += "Encoding ";
    if (flags & AV_OPT_FLAG_DEPRECATED)
        result += "Deprecated ";
    if (flags & AV_OPT_FLAG_READONLY)
        result += "ReadOnly ";
    // Add more flags as needed
    return result.empty() ? "None" : result;
}

// Helper function to convert AVRational to string
static std::string rational_to_string(const AVRational& r)
{
    CELUX_INFO("Starting rational_to_string with AVRational: {}/{}", r.num, r.den);
    std::ostringstream oss;
    if (r.den == 0)
    {
        oss << "N/A";
    }
    else
    {
        double value = static_cast<double>(r.num) / r.den;
        oss << value;
    }
    return oss.str();
}

// Helper function to extract default value as string
static std::string get_option_default(const AVOption* option)
{
    CELUX_INFO("Starting get_option_default with option: {}", option->name);
    if (!option)
    {
        return "None";
    }

    std::ostringstream oss;

    // Determine if the option has a default value
    bool has_default = true;

    switch (option->type)
    {
    case AV_OPT_TYPE_STRING:
    case AV_OPT_TYPE_COLOR:
    case AV_OPT_TYPE_CHLAYOUT:
    case AV_OPT_TYPE_BINARY:
    case AV_OPT_TYPE_DICT:
    case AV_OPT_TYPE_IMAGE_SIZE:
        has_default =
            (option->default_val.str != nullptr && option->default_val.str[0] != '\0');
        if (has_default)
            oss << option->default_val.str;
        else
            oss << "No Default";
        break;

    case AV_OPT_TYPE_INT:
    case AV_OPT_TYPE_INT64:
    case AV_OPT_TYPE_UINT64:
    case AV_OPT_TYPE_FLAGS:
    case AV_OPT_TYPE_DURATION:
        oss << option->default_val.i64;
        break;

    case AV_OPT_TYPE_DOUBLE:
    case AV_OPT_TYPE_FLOAT:
        oss << std::fixed << std::setprecision(2) << option->default_val.dbl;
        break;

    case AV_OPT_TYPE_BOOL:
        oss << (option->default_val.i64 ? "true" : "false");
        break;

    case AV_OPT_TYPE_RATIONAL:
    case AV_OPT_TYPE_VIDEO_RATE:
        if (option->default_val.q.num != 0 || option->default_val.q.den != 0)
            oss << rational_to_string(option->default_val.q);
        else
            oss << "No Default";
        break;

    case AV_OPT_TYPE_SAMPLE_FMT:
    {
        AVSampleFormat fmt = static_cast<AVSampleFormat>(option->default_val.i64);
        if (fmt == AV_SAMPLE_FMT_NONE)
        {
            oss << "No Default";
        }
        else
        {
            const char* sampleName = av_get_sample_fmt_name(fmt);
            oss << (sampleName ? sampleName : "Unknown Format");
        }
    }
    break;

    case AV_OPT_TYPE_PIXEL_FMT:
    {
        AVPixelFormat fmt = static_cast<AVPixelFormat>(option->default_val.i64);
        if (fmt == AV_PIX_FMT_NONE)
        {
            oss << "No Default";
        }
        else
        {
            const char* name = av_get_pix_fmt_name(fmt);
            oss << (name ? name : "Unknown Format");
        }
    }
    break;

    case AV_OPT_TYPE_CONST:
        // Constants represent possible values, default is their value
        oss << option->default_val.i64;
        break;

    default:
        oss << "Unsupported Type";
        break;
    }

    return oss.str();
}

// Function to determine if an option is required based on the absence of a default
// value
static bool is_option_required(const AVOption* opt)
{
    CELUX_INFO("Starting is_option_required with option: {}", opt->name);

    // Exclude deprecated options
    if (opt->flags & AV_OPT_FLAG_DEPRECATED)
        return false;

    switch (opt->type)
    {
    case AV_OPT_TYPE_STRING:
    case AV_OPT_TYPE_COLOR:
    case AV_OPT_TYPE_CHLAYOUT:
    case AV_OPT_TYPE_BINARY:
    case AV_OPT_TYPE_DICT:
    case AV_OPT_TYPE_IMAGE_SIZE:
        // Required if default string is nullptr or empty
        return (opt->default_val.str == nullptr || opt->default_val.str[0] == '\0');

    case AV_OPT_TYPE_INT:
    case AV_OPT_TYPE_INT64:
    case AV_OPT_TYPE_UINT64:
    case AV_OPT_TYPE_FLAGS:
        // Adjust based on whether a sentinel value indicates no default
        // For now, assume optional unless a special value indicates required
        return false;

    case AV_OPT_TYPE_DOUBLE:
    case AV_OPT_TYPE_FLOAT:
        // Zero might be a valid default; assume optional
        return false;

    case AV_OPT_TYPE_BOOL:
        // Booleans typically have a default value; assume optional
        return false;

    case AV_OPT_TYPE_RATIONAL:
    case AV_OPT_TYPE_VIDEO_RATE:
        // Required if numerator and denominator are zero
        return (opt->default_val.q.num == 0 && opt->default_val.q.den == 0);

    case AV_OPT_TYPE_DURATION:
        // Required if default is AV_NOPTS_VALUE
        return (opt->default_val.i64 == AV_NOPTS_VALUE);

    case AV_OPT_TYPE_PIXEL_FMT:
        // Required if default is AV_PIX_FMT_NONE
        return (opt->default_val.i64 == AV_PIX_FMT_NONE);

    case AV_OPT_TYPE_SAMPLE_FMT:
        // Required if default is AV_SAMPLE_FMT_NONE
        return (opt->default_val.i64 == AV_SAMPLE_FMT_NONE);

    case AV_OPT_TYPE_CONST:
    case AV_OPT_TYPE_FLAG_ARRAY:
        // Not user-settable or assume optional
        return false;

    default:
        // Unhandled types are assumed optional
        return false;
    }
}

static void list_ffmpeg_filters(const std::string& output_filename)
{
    CELUX_INFO("Starting list_ffmpeg_filters with output file: {}", output_filename);

    // Create a JSON array to hold all filters
    json filters_json = json::array();

    // Create an opaque pointer for iteration
    void* opaque = nullptr;
    const AVFilter* filter = nullptr;

    CELUX_INFO("Starting to iterate over all filters");

    // Iterate over all available filters
    while ((filter = av_filter_iterate(&opaque)))
    {
        if (!filter)
        {
            CELUX_INFO("No more filters to iterate");
            break;
        }

        const char* filter_name = filter->name;
        const char* filter_desc = filter->description;
        CELUX_INFO("Processing filter: {}", filter_name);

        // Create a JSON object for this filter
        json filter_json;
        filter_json["filter_name"] = filter_name;
        filter_json["description"] =
            filter_desc ? filter_desc : "No description available";

        // Create an array to hold the options
        json options_json = json::array();

        // Check if the filter has a priv_class
        if (filter->priv_class)
        {
            CELUX_INFO("Filter '{}' has priv_class", filter_name);

            // Collect all options into a vector
            std::vector<const AVOption*> options;
            const AVOption* option = nullptr;
            while ((option = av_opt_next(&filter->priv_class, option)))
            {
                options.push_back(option);
            }

            // Build a mapping from unit names to constants (aliases)
            std::unordered_map<std::string, std::vector<const AVOption*>>
                unit_to_constants;
            for (const AVOption* opt : options)
            {
                if (opt->type == AV_OPT_TYPE_CONST && opt->unit)
                {
                    unit_to_constants[opt->unit].push_back(opt);
                }
            }

            // Process each option
            for (const AVOption* opt : options)
            {
                if (opt->type == AV_OPT_TYPE_CONST)
                {
                    // Skip constants here; they are used in possible_values
                    continue;
                }

                CELUX_INFO("Processing option '{}' for filter '{}'", opt->name,
                           filter_name);

                // Create a JSON object for this option
                json option_json;
                option_json["name"] = opt->name;
                option_json["help"] = opt->help ? opt->help : "";
                option_json["type"] = option_type_to_string(opt->type);
                option_json["default"] = get_option_default(opt);
                option_json["required"] = is_option_required(opt);
                option_json["flags"] = option_flags_to_string(opt->flags);
                option_json["flags_value"] = opt->flags;
                option_json["min"] = opt->min;
                option_json["max"] = opt->max;
                option_json["unit"] = opt->unit ? opt->unit : "";
                option_json["deprecated"] = (opt->flags & AV_OPT_FLAG_DEPRECATED) != 0;
                option_json["readonly"] = (opt->flags & AV_OPT_FLAG_READONLY) != 0;

                // Include possible values if available (aliases)
                if (opt->unit)
                {
                    auto it = unit_to_constants.find(opt->unit);
                    if (it != unit_to_constants.end())
                    {
                        const auto& constants = it->second;
                        json possible_values = json::array();
                        for (const AVOption* const_opt : constants)
                        {
                            json const_json;
                            const_json["name"] = const_opt->name;
                            const_json["value"] = get_option_default(const_opt);
                            const_json["help"] = const_opt->help ? const_opt->help : "";
                            possible_values.push_back(const_json);
                        }
                        option_json["possible_values"] = possible_values;
                    }
                }

                // Add the option to the options array
                options_json.push_back(option_json);
            }
        }
        else
        {
            CELUX_INFO("Filter '{}' does not have priv_class", filter_name);
        }

        // Add the options array to the filter JSON object
        filter_json["options"] = options_json;

        // Add the filter JSON object to the filters array
        filters_json.push_back(filter_json);
    }

    CELUX_INFO("Completed iterating over all filters");

    // Serialize the JSON object to a string
    std::string json_output = filters_json.dump(4); // 4-space indentation

    // Write the JSON string to the output file
    std::ofstream outfile(output_filename);
    if (!outfile.is_open())
    {
        CELUX_INFO("Failed to open file: {}", output_filename);
        return;
    }

    outfile << json_output;
    outfile.close();

    CELUX_INFO("Filter information has been written to {}", output_filename);
    std::cout << "Filter information has been written to " << output_filename
              << std::endl;
}


namespace celux
{
/**
 * @brief Utility function to get a suitable hardware configuration for a codec.
 *
 * @param codec A pointer to the codec for which hardware configuration is needed.
 * @return A pointer to the suitable AVCodecHWConfig, or nullptr if none found.
 */
inline const AVCodecHWConfig* getSuitableHWConfig(const AVCodec* codec)
{
    const AVCodecHWConfig* hwConfig = nullptr;
    int index = 0;
    while ((hwConfig = avcodec_get_hw_config(codec, index++)))
    {
        // Check if the configuration supports hardware device context
        if (hwConfig->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)
        {
            return hwConfig;
        }
    }
    return nullptr; // No suitable hardware configuration found
}

/**
 * @brief Utility function to log supported hardware configurations for a codec.
 *
 * @param codec A pointer to the codec for which hardware configurations are to be
 * logged.
 */
inline void logSupportedHardwareConfigs(const AVCodec* codec)
{
    const AVCodecHWConfig* hwConfig = nullptr;
    int index = 0;
    while ((hwConfig = avcodec_get_hw_config(codec, index++)))
    {
        // Get the device type name for the hardware configuration
        const char* deviceTypeName = av_hwdevice_get_type_name(hwConfig->device_type);
        if (deviceTypeName)
        {
            std::cout << "Supported hardware config: " << deviceTypeName << std::endl;
        }
    }
}

/**
 * @brief Utility function to convert FFmpeg error codes to readable strings.
 *
 * @param errorCode The FFmpeg error code.
 * @return A string representation of the error.
 */
inline std::string errorToString(int errorCode)
{
    char errBuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(errorCode, errBuf, AV_ERROR_MAX_STRING_SIZE);
    return std::string(errBuf);
}

/**
 * @brief Checks if hardware acceleration is supported for a given codec.
 *
 * @param codec A pointer to the codec to check for hardware acceleration support.
 * @return True if hardware acceleration is supported, false otherwise.
 */
inline bool isHardwareAccelerationSupported(const AVCodec* codec)
{
    // Iterate over all hardware configurations for the codec
    for (int i = 0;; i++)
    {
        const AVCodecHWConfig* hwConfig = avcodec_get_hw_config(codec, i);
        if (!hwConfig)
        {
            break; // No more configurations
        }

        // Check if the codec has any hardware acceleration capabilities
        if (hwConfig->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)
        {
            return true;
        }
    }
    return false;
}

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
struct AVFilterGraphDeleter
{
    void operator()(AVFilterGraph* graph) const
    {
        avfilter_graph_free(&graph);
    }
};
// SwrContext* swrCtx = nullptr;
struct SwrContextDeleter
{
    void operator()(SwrContext* ctx) const
    {
		swr_free(&ctx);
	}
};

using AVFormatContextPtr = std::unique_ptr<AVFormatContext, AVFormatContextDeleter>;
    using AVCodecContextPtr = std::unique_ptr<AVCodecContext, AVCodecContextDeleter>;
    using AVBufferRefPtr = std::unique_ptr<AVBufferRef, AVBufferRefDeleter>;
    using AVPacketPtr = std::unique_ptr<AVPacket, AVPacketDeleter>;
    using AVFilterGraphPtr = std::unique_ptr<AVFilterGraph, AVFilterGraphDeleter>;
    using SwrContextPtr = std::unique_ptr<SwrContext, SwrContextDeleter>;

} // namespace celux

#endif // CX_CORE_HPP