#ifndef VIDEOREADER_HPP
#define VIDEOREADER_HPP

#include "Decoder.hpp" // Ensure this includes the Filter class
#include "Factory.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <VideoEncoder.hpp>

namespace py = pybind11;

/**
 * @brief Enum representing the output backend type.
 */
enum class Backend
{
    PyTorch, // Return frames as torch::Tensor (default)
    NumPy    // Return frames as numpy.ndarray
};

class VideoReader
{
  public:
    /**
     * @brief Constructs a VideoReader for a given input file.
     *
     * @param filePath Path to the input video file.
     * @param numThreads Number of threads to use for decoding.
     * @param force_8bit Force 8-bit output regardless of source bit depth.
     * @param backend Output backend type ("pytorch" or "numpy"). Default is "pytorch".
     * @param decode_accelerator Decode acceleration type ("cpu" or "nvdec"). Default is "cpu".
     * @param cuda_device_index CUDA device index for NVDEC (default: 0).
     */
    VideoReader(const std::string& filePath,
                int numThreads = static_cast<int>(std::thread::hardware_concurrency() / 2),
                bool force_8bit = false,
                Backend backend = Backend::PyTorch,
                const std::string& decode_accelerator = "cpu",
                int cuda_device_index = 0);

    /**
     * @brief Destructor for VideoReader.
     */
    ~VideoReader();

    /**
     * @brief Create a VideoEncoder configured to this reader's video & audio
     * properties.
     * @param outputPath Path where the new file will be saved.
     * @return Shared pointer to a VideoEncoder pre-configured for resolution, fps, and
     * audio.
     */
    std::shared_ptr<celux::VideoEncoder> createEncoder(const std::string& outputPath) const;

    // Direct property getters for performance
    int getWidth() const { return properties.width; }
    int getHeight() const { return properties.height; }
    double getFps() const { return properties.fps; }
    double getMinFps() const { return properties.min_fps; }
    double getMaxFps() const { return properties.max_fps; }
    double getDuration() const { return properties.duration; }
    int getTotalFrames() const { return properties.totalFrames; }
    bool getHasAudio() const { return properties.hasAudio; }
    int getAudioBitrate() const { return properties.audioBitrate; }
    int getAudioChannels() const { return properties.audioChannels; }
    int getAudioSampleRate() const { return properties.audioSampleRate; }
    std::string getAudioCodec() const { return properties.audioCodec; }
    int getBitDepth() const { return properties.bitDepth; }
    double getAspectRatio() const { return properties.aspectRatio; }
    std::string getCodec() const { return properties.codec; }
    std::string getPixelFormat() const;
    
    /**
     * @brief Get the properties of the video.
     *
     * @return py::dict Dictionary containing video properties.
     */
    py::dict getProperties() const;

    /**
     * @brief Read a frame from the video.
     *
     * Depending on the configuration, returns either a torch::Tensor or a
     * numpy.ndarray. Shape is always HWC.
     *
     * @return py::object The next frame (torch::Tensor or numpy.ndarray based on backend).
     */
    py::object readFrame();

    /**
     * @brief Internal method to decode the next frame into the internal tensor buffer.
     *
     * @return torch::Tensor The decoded frame as torch::Tensor (internal use).
     */
    torch::Tensor decodeFrame();

    /**
     * @brief Seek to a specific timestamp in the video.
     *
     * @param timestamp The timestamp to seek to (in seconds).
     * @return true if seek was successful, false otherwise.
     */
    bool seek(double timestamp);

    /**
     * @brief Get a list of supported codecs.
     *
     * @return std::vector<std::string> List of supported codec names.
     */
    std::vector<std::string> supportedCodecs();

    /**
     * @brief Overloads the [] operator to access video properties by key.
     *
     * @param key The property key to access.
     * @return py::object The value associated with the key.
     */
    py::object operator[](py::object key);
    /**
     * @brief Get the total number of frames.
     *
     * @return int Total frame count.
     */
    int length() const;

    /**
     * @brief Set the range of frames or timestamps to read.
     *
     * If the start and end values are integers, they are interpreted as frame numbers.
     * If they are floating-point numbers, they are interpreted as timestamps in
     * seconds.
     *
     * @param start Starting frame number or timestamp.
     * @param end Ending frame number or timestamp.
     */
    void setRange(std::variant<int, double> start, std::variant<int, double> end);

    /**
     * @brief Add a filter to the decoder's filter pipeline.
     *
     * @param filterName Name of the filter (e.g., "scale").
     * @param filterOptions Options for the filter (e.g., "1280:720").
     */
    void addFilter(const std::string& filterName, const std::string& filterOptions);

    /**
     * @brief Initialize the decoder after adding all desired filters.
     *
     * This separates filter addition from decoder initialization, allowing
     * users to configure filters before starting the decoding process.
     *
     * @return true if initialization is successful.
     * @return false otherwise.
     */
    bool initialize();
    /**
     * @brief Set the range of frames to read (helper function).
     *
     * @param startFrame Starting frame index.
     * @param endFrame Ending frame index (-1 for no limit).
     */
    void setRangeByFrames(int startFrame, int endFrame);

    /**
     * @brief Set the range of timestamps to read (helper function).
     *
     * @param startTime Starting timestamp.
     * @param endTime Ending timestamp (-1 for no limit).
     */
    void setRangeByTimestamps(double startTime, double endTime);

    
    /**
     * @brief Retrieve the audio object for interaction.
     *
     * @return Audio instance.
     */
    class Audio
    {
      public:
        explicit Audio(std::shared_ptr<celux::Decoder> decoder);

        /**
         * @brief Get audio data as a tensor.
         *
         * @return torch::Tensor The extracted audio data.
         */
        torch::Tensor getAudioTensor();

        /**
         * @brief Extracts the audio to a file.
         *
         * @param outputFilePath The path where the audio file should be saved.
         * @return True if successful, false otherwise.
         */
        bool extractToFile(const std::string& outputFilePath);

        /**
         * @brief Get audio properties such as sample rate, channels, and codec.
         *
         * @return A struct containing audio properties.
         */
        celux::Decoder::VideoProperties getProperties() const;

      private:
        std::shared_ptr<celux::Decoder> decoder;
    };

    /**
     * @brief Get the audio interface.
     *
     * @return A reference to the Audio class.
     */
    std::shared_ptr<Audio> getAudio();
    /**
     * @brief Get the frame at (or immediately after) a timestamp, in seconds.
     *        Uses a secondary decoder; does not disturb sequential iteration.
     * @param timestamp_seconds Timestamp in seconds (0 <= t <= duration).
     * @return py::object HWC frame (torch::Tensor or numpy.ndarray based on backend).
     * @throws std::out_of_range on invalid timestamp, std::runtime_error on failure.
     */
    py::object frameAt(double timestamp_seconds);

    /**
     * @brief Get the frame at (or immediately after) a frame index.
     *        Uses a secondary decoder; does not disturb sequential iteration.
     * @param frame_index 0-based frame index (0 <= idx < total_frames).
     * @return py::object HWC frame (torch::Tensor or numpy.ndarray based on backend).
     * @throws std::out_of_range on invalid index, std::runtime_error on failure.
     */
    py::object frameAt(int frame_index);

    /**
     * @brief Iterator support: returns self.
     */
    VideoReader& iter();

    /**
     * @brief Iterator support: returns next frame or throws StopIteration.
     * @return py::object HWC frame (torch::Tensor or numpy.ndarray based on backend).
     */
    py::object next();

    /**
     * @brief Context manager enter.
     */
    void enter();

    /**
     * @brief Context manager exit.
     */
    void exit(const py::object& exc_type, const py::object& exc_value, const py::object& traceback);

    /**
     * @brief Reset the reader to the beginning.
     */
    void reset();

    /**
     * @brief Get the frame count from metadata (no pre-scanning).
     * @return int64_t Total number of frames in the video.
     */
    int64_t getFrameCount() const;

    /**
     * @brief Decode a batch of frames at specified indices.
     * @param indices Vector of frame indices to decode.
     * @return torch::Tensor Batch tensor of shape [B, H, W, C].
     */
    torch::Tensor decodeBatch(const std::vector<int64_t>& indices);

  private:
    void ensureRandDecoder();
    bool seekToFrame(int frame_number);
    torch::ScalarType findTypeFromBitDepth();
    double frameDuration() const
    {
        return (properties.fps > 0.0) ? 1.0 / properties.fps : 0.0;
    }
    std::shared_ptr<celux::Decoder> rand_decoder;

    torch::Tensor makeLikeOutputTensor() const;
    /**
     * @brief Close the video reader and release resources.
     */
    void close();

    /**
     * @brief Convert a torch::Tensor to a py::object based on the backend setting.
     *
     * @param t The torch::Tensor to convert.
     * @return py::object Either the tensor or a numpy array.
     */
    py::object tensorToOutput(const torch::Tensor& t) const;

    /**
     * @brief Internal method to decode frame at timestamp using rand_decoder.
     *
     * @param timestamp_seconds Timestamp in seconds.
     * @return torch::Tensor The decoded frame.
     */
    torch::Tensor decodeFrameAt(double timestamp_seconds);

    /**
     * @brief Internal method to decode frame at index using rand_decoder.
     *
     * @param frame_index Frame index.
     * @return torch::Tensor The decoded frame.
     */
    torch::Tensor decodeFrameAt(int frame_index);

    // Member variables
    std::shared_ptr<celux::Decoder> decoder;
    celux::Decoder::VideoProperties properties;

    torch::Tensor tensor;

    // Variables for frame range
    int start_frame = 0;
    int end_frame = -1; // -1 indicates no limit

    // Variables for timestamp range
    double start_time = 0.0;
    double end_time = -1.0; // -1 indicates no limit

    // Iterator state
    int currentIndex;
    double current_timestamp; // Add this line
    // List of filters to be added before initialization
    torch::Tensor bufferedFrame; // The "first valid" frame, if we found it early
    bool hasBufferedFrame = false;
    std::shared_ptr<Audio> audio;

    // Lazy loading support
    std::string filePath;
    int numThreads;
    bool force_8bit = false;
    Backend backend = Backend::PyTorch; // Output backend selection
    celux::DecodeAccelerator decodeAccelerator = celux::DecodeAccelerator::CPU;
    int cudaDeviceIndex = 0;
};

#endif // VIDEOREADER_HPP
