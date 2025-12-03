[![Release and Benchmark Tests](https://github.com/Trentonom0r3/CeLux/actions/workflows/createRelease.yaml/badge.svg)](https://github.com/Trentonom0r3/CeLux/actions/workflows/createRelease.yaml)
[![License](https://img.shields.io/badge/license-AGPL%203.0-blue.svg)](https://github.com/Trentonom0r3/CeLux/blob/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/nelux)](https://pypi.org/project/nelux/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/nelux)](https://pypi.org/project/nelux/)
[![Python Versions](https://img.shields.io/pypi/pyversions/nelux)](https://pypi.org/project/nelux/)
[![Discord](https://img.shields.io/discord/1041502781808328704.svg?label=Join%20Us%20on%20Discord&logo=discord&colorB=7289da)](https://discord.gg/hFSHjGyp4p)
# CeLux

Originally created by [Trentonom0r3](https://github.com/Trentonom0r3)

**CeLux** is a high‑performance Python library for video processing, leveraging the power of libav(FFmpeg). It delivers some of the fastest decode times for full‑HD videos globally, enabling efficient video decoding directly into PyTorch tensors—and now simplified, one‑call audio muxing straight from a tensor. At present, CeLux offers ***limited, but functional*** encoding support.

The name **CeLux** comes from the Latin words _celer_ (speed) and _lux_ (light), reflecting its commitment to speed and efficiency.

### Example
```python
from celux import VideoReader

vr = VideoReader("input.mp4")

frame_ts = vr.frame_at(12.34)   # by timestamp
frame_idx = vr.frame_at(1000)   # by frame index

print(frame_ts.shape, frame_ts.dtype)
print(frame_idx.shape, frame_idx.dtype)
```


## 📚 Documentation

- [📝 Changelog](https://github.com/Trentonom0r3/CeLux/blob/master/docs/CHANGELOG.md)
- [🍎 Audio & Muxing Guide](https://github.com/Trentonom0r3/CeLux/blob/master/docs/FAQ.md#audio)
- [📊 Benchmarks](https://github.com/NevermindNilas/python-decoders-benchmarks/blob/main/1280x720_diagram.png)


## 🚀 Features

- ⚡ **Ultra‑Fast Video Decoding:** Lightning‑fast decode times for full‑HD videos using hardware acceleration.
- 🔗 **Direct Decoding to Tensors:** Frames come out as PyTorch tensors (`HWC` layout by default).
- 🔊 **Simplified Audio Encoding:** One call to `encode_audio_tensor()` streams raw PCM into the encoder.
- 🔄 **Easy Integration:** Drop‑in replacement for your existing Python + PyTorch workflows.

### Q: How do I report a bug or request a feature?

**A:** Open an issue on our [GitHub Issues](https://github.com/Trentonom0r3/celux/issues) with as much detail as you can (FFmpeg version, platform, repro steps, etc.).


## ⚡ Quick Start

```bash
pip install celux
```

**FOR LINUX**
- Download the most recent release (.whl)

```bash
pip install ./*.whl
```

```python
from celux import VideoReader
import torch

reader = VideoReader("/path/to/input.mp4")
with reader.create_encoder("/path/to/output.mp4") as enc:
    # 1) Re‑encode video frames
    for frame in reader:
        enc.encode_frame(frame)

    # 2) If there’s audio, hand off the entire PCM in one go:
    if reader.has_audio:
        pcm = reader.audio.tensor().to(torch.int16)
        enc.encode_audio_frame(pcm)

print("Done!")
```

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[FFmpeg](https://ffmpeg.org/):** The backbone of video processing in CeLux.
- **[PyTorch](https://pytorch.org/):** For tensor operations and CUDA support.
- **[Vcpkg](https://github.com/microsoft/vcpkg):** Simplifies cross‑platform dependency management.
- **[@NevermindNilas](https://github.com/NevermindNilas):** For assistance with testing, API suggestions, and more.

## 🚤 Roadmap

- **Support for Additional Codecs:**  
  - Expand hardware‑accelerated decoding/muxing support to VP9, AV1, etc.  
- **Audio Filters & Effects:**  
  - Add simple audio‑only filters (gain, resample, stereo panning).  
- **Advanced Muxing Options:**  
  - Expose more container parameters (subtitle tracks, chapters).  
- **Cross‑Platform CI:**  
  - Ensure Windows, macOS, Linux builds all pass full audio+video tests.
    (My current focus is windows, would love help getting linux side working as well!)
