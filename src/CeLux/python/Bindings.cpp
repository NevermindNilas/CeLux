#include "Python/VideoReader.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <filter_bindings.hpp>
#include <torch/extension.h>

namespace py = pybind11;
#define PYBIND11_DETAILED_ERROR_MESSAGES

PYBIND11_MODULE(celux, m)
{

    // Register the filters into the 'filters' submodule
    register_filters(m);
    // VideoReader bindings
    py::class_<VideoReader>(m, "VideoReader")
        .def(py::init<const std::string&, int, 
                      std::vector<std::shared_ptr<FilterBase>>, std::string&>(),
             py::arg("input_path"),
             py::arg("num_threads") =
                 static_cast<int>(std::thread::hardware_concurrency() / 2),
             py::arg("filters") = std::vector<std::shared_ptr<FilterBase>>(),
             py::arg("tensor_shape") = "HWC")
        .def("read_frame", &VideoReader::readFrame)
        .def_property_readonly("properties", &VideoReader::getProperties)
        .def_property_readonly("properties",
                               &VideoReader::getProperties) // Keep full dict access
        .def_property_readonly("width", [](const VideoReader& self)
                               { return self.getProperties()["width"].cast<int>(); })
        .def_property_readonly("height", [](const VideoReader& self)
                               { return self.getProperties()["height"].cast<int>(); })
        .def_property_readonly("fps", [](const VideoReader& self)
                               { return self.getProperties()["fps"].cast<double>(); })
        .def_property_readonly(
            "min_fps", [](const VideoReader& self)
            { return self.getProperties()["min_fps"].cast<double>(); })
        .def_property_readonly(
            "max_fps", [](const VideoReader& self)
            { return self.getProperties()["max_fps"].cast<double>(); })
        .def_property_readonly(
            "duration", [](const VideoReader& self)
            { return self.getProperties()["duration"].cast<double>(); })
        .def_property_readonly(
            "total_frames", [](const VideoReader& self)
            { return self.getProperties()["total_frames"].cast<int>(); })
        .def_property_readonly(
            "pixel_format", [](const VideoReader& self)
            { return self.getProperties()["pixel_format"].cast<std::string>(); })
        .def_property_readonly(
            "has_audio", [](const VideoReader& self)
            { return self.getProperties()["has_audio"].cast<bool>(); })
        .def_property_readonly(
            "audio_bitrate", [](const VideoReader& self)
            { return self.getProperties()["audio_bitrate"].cast<int>(); })
        .def_property_readonly(
            "audio_channels", [](const VideoReader& self)
            { return self.getProperties()["audio_channels"].cast<int>(); })
        .def_property_readonly(
            "audio_sample_rate", [](const VideoReader& self)
            { return self.getProperties()["audio_sample_rate"].cast<int>(); })
        .def_property_readonly(
            "audio_codec", [](const VideoReader& self)
            { return self.getProperties()["audio_codec"].cast<std::string>(); })
        .def_property_readonly("bit_depth",
                               [](const VideoReader& self) {
                                   return self.getProperties()["bit_depth"].cast<int>();
                               })
        .def_property_readonly(
            "aspect_ratio", [](const VideoReader& self)
            { return self.getProperties()["aspect_ratio"].cast<double>(); })
        .def_property_readonly(
            "codec", [](const VideoReader& self)
            { return self.getProperties()["codec"].cast<std::string>(); })
        .def_property_readonly("audio", &VideoReader::getAudio)
        .def("supported_codecs", &VideoReader::supportedCodecs)
        .def("get_properties", &VideoReader::getProperties)
        .def("__getitem__", &VideoReader::operator[])
        .def("__len__", &VideoReader::length)
        .def(
            "__iter__", [](VideoReader& self) -> VideoReader& { return self.iter(); },
            py::return_value_policy::reference_internal)
        .def("__next__", &VideoReader::next)
        .def(
            "__enter__",
            [](VideoReader& self) -> VideoReader&
            {
                self.enter();
                return self;
            },
            py::return_value_policy::reference_internal)
        .def("__exit__", &VideoReader::exit)
        .def("reset", &VideoReader::reset)
        .def("set_range", &VideoReader::setRange, py::arg("start"), py::arg("end"),
             "Set the range using either frame numbers (int) or timestamps (float).")
        .def(
            "__call__",
            [](VideoReader& self, py::object arg) -> VideoReader&
            {
                if (py::isinstance<py::list>(arg) || py::isinstance<py::tuple>(arg))
                {
                    auto range_list = arg.cast<std::vector<py::object>>();
                    if (range_list.size() != 2)
                    {
                        throw std::runtime_error(
                            "Range must be a list or tuple of two elements");
                    }

                    py::object start_obj = range_list[0];
                    py::object end_obj = range_list[1];

                    // ----------------------------
                    // If both are ints => frames
                    // ----------------------------
                    if (py::isinstance<py::int_>(start_obj) &&
                        py::isinstance<py::int_>(end_obj))
                    {
                        int start = start_obj.cast<int>();
                        int end = end_obj.cast<int>();
                        // Call the *frame-based* method
                        self.setRangeByFrames(start, end);
                    }
                    // --------------------------------
                    // If both are floats => timestamps
                    // --------------------------------
                    else if (py::isinstance<py::float_>(start_obj) &&
                             py::isinstance<py::float_>(end_obj))
                    {
                        double start = start_obj.cast<double>();
                        double end = end_obj.cast<double>();
                        self.setRangeByTimestamps(start, end);
                    }
                    else
                    {
                        throw std::runtime_error(
                            "Start and end must both be int or both be float");
                    }
                }
                else
                {
                    throw std::runtime_error(
                        "Argument must be a list or tuple of two elements");
                }
                return self;
            },
            py::return_value_policy::reference_internal)
    // -------------------
    // Bind getAudio()
    // -------------------
    .def("get_audio", &VideoReader::getAudio,
         py::return_value_policy::reference_internal, "Retrieve the Audio object");

    // -------------------
    // Bind Audio Class
    // -------------------
    py::class_<VideoReader::Audio, std::shared_ptr<VideoReader::Audio>>(m, "Audio")
        .def(py::init<std::shared_ptr<celux::Decoder>>())
        .def("tensor", &VideoReader::Audio::getAudioTensor,
             "Retrieve audio as a PyTorch tensor")
        .def("file", &VideoReader::Audio::extractToFile,
             py::arg("output_path"), "Extract audio to a specified file path")
        .def_property_readonly("properties", &VideoReader::Audio::getProperties)
        .def_property_readonly("sample_rate", [](const VideoReader::Audio& self)
                               { return self.getProperties().audioSampleRate; })
        .def_property_readonly("channels", [](const VideoReader::Audio& self)
                               { return self.getProperties().audioChannels; })
        .def_property_readonly("bit_depth", [](const VideoReader::Audio& self)
                               { return self.getProperties().bitDepth; })
        .def_property_readonly("codec", [](const VideoReader::Audio& self)
                               { return self.getProperties().audioCodec; })
        .def_property_readonly("bitrate", [](const VideoReader::Audio& self)
                               { return self.getProperties().audioBitrate; });


    py::enum_<spdlog::level::level_enum>(m, "LogLevel")
        .value("trace", spdlog::level::trace)
        .value("debug", spdlog::level::debug)
        .value("info", spdlog::level::info)
        .value("warn", spdlog::level::warn)
        .value("error", spdlog::level::err)
        .value("critical", spdlog::level::critical)
        .value("off", spdlog::level::off)
        .export_values();

    m.def("set_log_level", &celux::Logger::set_level, "Set the logging level for CeLux",
          py::arg("level"));

    /*	enum class EncodingFormats
    {
        YUV420P, // 8 bit sw format -- requires input to be in Uint8, rgb24
                    // NV12, // 8 bit hw format
        YUV420P10LE, // 10 bit sw format -- requires input to be in Uint16, rgb48
       // P010, // 10 bit hw format
    };
    */
}
