// CPU Decoder.hpp
#pragma once

#include "backends/Decoder.hpp"

namespace celux::backends::cpu
{
class Decoder : public celux::Decoder
{
  public:
    Decoder(const std::string& filePath, int numThreads,
            std::vector<std::shared_ptr<FilterBase>> filters)
        : celux::Decoder( numThreads, filters)
    {
        initialize(filePath);
        initializeAudio();
    }

    // No need to override methods unless specific behavior is needed
};
} // namespace celux::backends::cpu
