#ifndef VIDEOREADER_HPP
#define VIDEOREADER_HPP

#include "Decoder.hpp" // Ensure this includes the Filter class
#include "Factory.hpp"
#include <pybind11/pybind11.h>


namespace py = pybind11;

class VideoReader
{
  public:
    /**
     * @brief Constructs a VideoReader for a given input file.
     *
     * @param filePath Path to the input video file.
     * @param numThreads Number of threads to use for decoding.
     * @param device Processing device ("cpu" or "cuda").
     */
    VideoReader(const std::string& filePath,
                int numThreads = static_cast<int>(std::thread::hardware_concurrency() /
                                                  2),
                std::vector<std::shared_ptr<FilterBase>> filter = {}, std::string tensorShape = "CHW");

    /**
     * @brief Destructor for VideoReader.
     */
    ~VideoReader();

    /**
     * @brief Overloads the [] operator to access video properties by key.
     *
     * @param key The property key to access.
     * @return py::object The value associated with the key.
     */
    py::object operator[](py::object key);
    /**
     * @brief Read a frame from the video.
     *
     * Depending on the configuration, returns either a torch::Tensor or a
     * py::array<uint8_t>. Shape is always HWC. If batch size is specified in Reader
     * config, output shape will be BHWC for Tensors.
     *
     * @return torch::Tensor The next frame as torch::Tensor.
     */
    torch::Tensor readFrame();

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
     * @brief Get the properties of the video.
     *
     * @return py::dict Dictionary containing video properties.
     */
    py::dict getProperties() const;

    /**
     * @brief Reset the video reader to the beginning.
     */
    void reset();

    /**
     * @brief Iterator initialization for Python.
     *
     * @return VideoReader& Reference to the VideoReader object.
     */
    VideoReader& iter();

    /**
     * @brief Get the next frame in iteration.
     *
     * @return torch::Tensor Next frame as torch::Tensor.
     */
    torch::Tensor next();

    /**
     * @brief Enter method for Python context manager.
     */
    void enter();

    /**
     * @brief Exit method for Python context manager.
     *
     * @param exc_type Exception type (if any).
     * @param exc_value Exception value (if any).
     * @param traceback Traceback object (if any).
     */
    void exit(const py::object& exc_type, const py::object& exc_value,
              const py::object& traceback);

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

  private:
    bool seekToFrame(int frame_number);
    torch::ScalarType findTypeFromBitDepth();

    /**
     * @brief Close the video reader and release resources.
     */
    void close();

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
    std::vector<std::shared_ptr<FilterBase>> filters_;
    torch::Tensor bufferedFrame; // The "first valid" frame, if we found it early
    bool hasBufferedFrame = false;
    std::shared_ptr<Audio> audio;
};

#endif // VIDEOREADER_HPP
