#pragma once
#ifndef CPU_CONVERTERS_HPP
#define CPU_CONVERTERS_HPP

// Base classes and existing converters
#include <cpu/BGRAToRGB.hpp>
#include <cpu/BGRToRGB.hpp>
#include <cpu/CPUConverter.hpp>
#include <cpu/GBRPToRGB.hpp>
#include <cpu/RGBAToRGB.hpp>
#include <cpu/RGBToRGB.hpp>
#include <cpu/YUV420P10ToRGB48.hpp>
#include <cpu/YUV420PToRGB.hpp>
#include <cpu/YUV422P10ToRGB48.hpp>

// -------------------------------------------------------------------------
// New converters for additional pixel formats
// -------------------------------------------------------------------------

// 8-bit YUV422 -> RGB24
#include <cpu/YUV422P8ToRGB24.hpp>

// 12-bit YUV420 -> 48-bit RGB
#include <cpu/YUV420P12ToRGB48.hpp>

// 12-bit YUV422 -> 48-bit RGB
#include <cpu/YUV422P12ToRGB48.hpp>

// 8-bit YUV444 -> RGB24
#include <cpu/YUV444P8ToRGB24.hpp>

// 10-bit YUV444 -> 48-bit RGB
#include <cpu/YUV444P10ToRGB48.hpp>

// 12-bit YUV444 -> 48-bit RGB
#include <cpu/YUV444P12ToRGB48.hpp>

#endif // CPU_CONVERTERS_HPP
