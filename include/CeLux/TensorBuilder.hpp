#pragma once

#include <CxCore.hpp>

class TensorBuilder
{
  public:
    // Constructor that takes the format string
    TensorBuilder(const std::string& format) : format_(format)
    {
        parseFormat();
    }

    // Create a tensor with specified height, width, data type, and devicen
    void createTensor(int64_t height, int64_t width, torch::Dtype dtype = torch::kUInt8,
                      torch::Device device = torch::kCPU)
    {
        // Map dimension characters to sizes
        std::unordered_map<char, int64_t> dim_sizes;

        // Set default sizes for 'B' and 'C' if they are in the format
        if (format_.find('B') != std::string::npos)
        {
            dim_sizes['B'] = 1; // Default batch size
        }
        if (format_.find('C') != std::string::npos)
        {
            dim_sizes['C'] = 3; // Default channel size
        }

        // Set sizes for 'H' and 'W'
        if (format_.find('H') != std::string::npos)
        {
            dim_sizes['H'] = height;
        }
        else
        {
            throw std::invalid_argument("Format string must contain 'H' for height.");
        }

        if (format_.find('W') != std::string::npos)
        {
            dim_sizes['W'] = width;
        }
        else
        {
            throw std::invalid_argument("Format string must contain 'W' for width.");
        }

        // Build the dimensions vector in the order specified by the format string
        std::vector<int64_t> full_dims;
        for (char c : format_)
        {
            if (dim_sizes.find(c) == dim_sizes.end())
            {
                throw std::invalid_argument("Dimension '" + std::string(1, c) +
                                            "' size not specified.");
            }
            full_dims.push_back(dim_sizes[c]);
        }

        // Create the tensor with the specified dimensions, data type, and device
        tensor_ = torch::empty(full_dims, torch::TensorOptions(dtype).device(device));
    }

    // Access the data pointer
    void* dataPtr()
    {
        return tensor_.data_ptr();
    }

    // Access the tensor itself
    at::Tensor& getTensor()
    {
        return tensor_;
    }

    // Convert tensor to a standard format (default is "BCHW")
    void convertToStandardFormat(const std::string& standardFormat = "BCHW")
    {
        if (standardFormat.size() != format_.size())
        {
            throw std::invalid_argument(
                "Standard format length must match tensor format length.");
        }

        // Determine the permutation order
        std::vector<int64_t> permuteOrder;
        for (char c : standardFormat)
        {
            if (dimOrder_.find(c) == dimOrder_.end())
            {
                throw std::invalid_argument("Dimension '" + std::string(1, c) +
                                            "' not found in tensor format.");
            }
            permuteOrder.push_back(dimOrder_[c]);
        }

        // Permute the tensor dimensions to match the standard format
        tensor_ = tensor_.permute(permuteOrder);

        // Update the format and dimension order to reflect the new layout
        format_ = standardFormat;
        parseFormat();
    }

  private:
    void parseFormat()
    {
        dimOrder_.clear();
        for (size_t i = 0; i < format_.size(); ++i)
        {
            char c = format_[i];
            if (dimOrder_.count(c))
            {
                throw std::invalid_argument("Duplicate dimension in format string: " +
                                            std::string(1, c));
            }
            dimOrder_[c] = i;
        }
    }

    std::string format_; // User-specified format
    at::Tensor tensor_;  // The tensor object
    std::unordered_map<char, int64_t>
        dimOrder_; // Mapping from format characters to dimension indices
};
