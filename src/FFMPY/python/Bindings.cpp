// bindings.cpp

#include "Python/VideoReader.hpp"
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(ffmpy, m)
{
    py::class_<VideoReader>(m, "VideoReader")
        .def(py::init<const std::string&, bool, const std::string&, bool, std::string&>(),
             py::arg("filePath"), py::arg("useHardware") = true,
             py::arg("hwType") = "cuda", py::arg("as_numpy") = false, py::arg("dtype") = "uint8")
        .def("readFrame", &VideoReader::readFrame)
        .def("seek", &VideoReader::seek)
        .def("supportedCodecs", &VideoReader::supportedCodecs)
        .def("getProperties", &VideoReader::getProperties)
        // Magic methods
        .def("__len__", &VideoReader::length)
        .def(
            "__iter__", [](VideoReader& self) -> VideoReader& { return self; },
            py::return_value_policy::reference_internal,
            "Return the iterator object itself.")
        .def("__next__", &VideoReader::next)
        .def(
            "__enter__",
            [](VideoReader& self) -> VideoReader&
            {
                self.enter();
                return self;
            },
            py::return_value_policy::reference_internal,
            "Enter the context manager and set 'as_numpy' flag.")

        .def("__exit__",
             [](VideoReader& self, py::object exc_type, py::object exc_value,
                py::object traceback)
             {
                 self.exit(exc_type, exc_value, traceback);
                 // Do not suppress exceptions
                 return false;
             });
}
