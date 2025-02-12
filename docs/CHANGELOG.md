## 📈 Changelog

### **Version 0.6.0 (2025-1-25)**  

#### **New Audio API in `VideoReader`**
#### **Retrieve Audio Data as a Tensor or File**

A new `.audio` property has been added to `VideoReader`, allowing direct access to the `Audio` object:

```python
reader = VideoReader("test.mp4")
if reader.has_audio:
    audio = reader.audio  # Access the audio object
    tensor = audio.tensor()  # Retrieve audio as a PyTorch tensor, 1D. NOTE. THIS HAS BEEN MINIMALLY TESTED
    success = audio.file("output.wav")  # Extract audio to a WAV file
```

#### **Audio Class Features**
- **`tensor()`** – Extracts the audio stream as a PyTorch tensor.
- **`file(output_path)`** – Saves the audio to a specified file path.
- **Read-only Properties:**
  - `sample_rate`: Audio sample rate in Hz.
  - `channels`: Number of audio channels.
  - `bit_depth`: Bit depth of the audio.
  - `codec`: Audio codec format.
  - `bitrate`: Audio bitrate.

Example usage:
```python
print(audio.sample_rate)  # Get the sample rate
print(audio.channels)  # Number of channels
```

### **Version 0.5.8.5 (2025-1-24)**  
- Fixed an issue with ranges being off by 1 frame.

### **Version 0.5.8 (2025-1-24)**  

#### **Improved Property Access for VideoReader**  
- Added direct property access for video metadata.  
- Users can now retrieve video properties directly instead of accessing `properties["key"]`.  
- Example:  
  ```python
  reader = VideoReader("test.mp4")
  print(reader.width)  # Instead of video.properties["width"]
  print(reader.fps)  # Instead of video.properties["fps"]
  ```

- **Available properties:**  
  `width, height, fps, min_fps, max_fps, duration, total_frames, pixel_format, has_audio, audio_bitrate, audio_channels, audio_sample_rate, audio_codec, bit_depth, aspect_ratio, codec`

#### **`__getitem__` Seeking Behavior Update**  
- `__getitem__` now only accepts **seconds (float)** for seeking.  
- Frame-based seeking (int) is currently **not supported** via `__getitem__`.  
- Example usage:  
  ```python
  reader = VideoReader("test.mp4")
  frame = reader[2.5]  # Seeks to 2.5 seconds into the video
  ```


### Version 0.5.7 (2025-1-23)

- **New color format support**  
  Added CPU-based conversions for:
  - 12-bit YUV (420, 422, and 444)
  - 10-bit YUV444
  - ProRes 4444 with alpha (10-bit or 12-bit)
  - Anything NOT 8-bit uses `uint-16` tensors.
  
- **Notes on rawvideo**  
  Raw `.yuv` files still require specifying resolution and pixel format manually.  
  (`-f rawvideo -pix_fmt yuv420p -s 1920x1080 …`)  
  - **IN PROGRESS****

- **Other improvements**  
  - Minor cleanups in converter classes.
  - Updated tests to cover the newly supported pixel formats.

### Version 0.5.6.1 (2025-1-23)
  = Adusted `__call__` method handling.
    - `Int` values seek as frames, `Float` values seek as times.
  - Added several tests against `OpenCV` and `FFMPEG` for confirmation on frame/time ranges.
  - Added tests for color formats.

### Version 0.5.6 (2025-1-22)
  = Removed `CUDA` Dependency in favor of CPU decoding. (It's faster anyways.)
  - Updated repo + docs
  - Tested and adjusted time range-- *should* match `OpenCV` behavior in all cases.
  = Looking into color space differences...

### Version 0.5.4 (2024-12-27)
  - Added Support for YUV422P10LE to RGB48 conversion. (CPU)
  - Added Support for GBRP Conversion. (CPU)
  - Added New Argument, `tensor_shape`
    - Default is `HWC`, but can be set to `CHW`, etc. 
    


### Version 0.5.3 (2024-11-10)
  - Fixed issue with floating point ranges.
  - Fixed/added quicktime and `RGB24` support.
  
### Version 0.5.2 (2024-11-05)
  - Finalized fixes for CPU Color Conversions.
  - Added Support for `RGBA`, `BGRA` pixel formats. 
  - Adjusted `__call__` and `set_range` methods for `VideoReader`.
    - Now takes `int` for frame steps, `float` for timestamp steps. 
      - Pass int or float and the reader will handle things internally.
  - No new benchmarks. Not need for this release.

### Version 0.5.1.2 (2024-11-05)
  - Fixed issue with Color Conversion.

### Version 0.5.1.1 (2024-11-05)
  - Testing out use of timestamps for setting range.

### Version 0.5.1 (2024-11-04)
  - Fixed an issue where if no filters were added, decoder would not run properly.

### Version 0.5.0 (2024-11-03)
  - Some Major Refactoring and changes made.
    - Parsed and created `Filter` classes for every (video) Ffmpeg filter.
      - Filters defined within `Celux.pyi`
        - Not all are tested. For Full documentation of arguments and usage, see: [ffmpeg-filter-docs](https://ffmpeg.org/ffmpeg-filters.html)
        - Please create a new issue if any problems occur!
    - Fixed an issue with Filter initialization and unwanted output messages. 
    ```py
    from celux import VideoReader, Scale #, CreateFilter, FilterType


    scale_filter = Scale(width = "1920", height = "1080")

    # scale_filter = CreateFilter(FilterType.Scale)
    # scale_filter.setWidth("1920")
    # scale_filter.setHeight"1080")
    # scale_filter.setFlags("bicubic")

    with VideoReader("input.mp4", device = "cpu", filters = [scale_filter]) as reader:
      for frame in reader:
        # will be a scaled frame
    ```


### Version 0.4.5.5 (2024-10-30)
  - Added some safety checks for filters.
    - Fixed issue that occurs when using `scale`.

### Version 0.4.5 (2024-10-29)
  - Implemented filters for `cpu` usage. 
    - usage should be familiar to those who've used `ffmpeg`:
  ```py  
  filters = [("scale", "1280:720"), ("hue", "0.5")]
  reader = cx.VideoReader("/path/to/input", device = "cpu", filters = filters)
  ```

### Version 0.4.4 (2024-10-29)
  - Removed Stream Parameter in `VideoReader`: The `VideoReader` no longer accepts an external CUDA stream. 
  - Introduced event-based synchronization between frame reading operations to ensure proper and consistent output.
  - Use of `nvdec` directly.

### Version 0.4.3.5 (2024-10-29)
  - Testing some changes, partial release, may end up reverting.
  - Use `nvdec` directly instead of `_cuvid`.
  - Some small refactoring and testing, nothing major.


### Version 0.4.3 (2024-10-29)
- **New Features**:
  - Added `num_threads` arg to control decoder threads internally used. 
  - Fixed `VideoReader()` calls, now properly sets frame range.
  - *Potentially* fixed issue with cuda synchronizations. 

### Version 0.4.2 (2024-10-28)
- **Focus on `VideoReader`**:
  - Removed `VideoWriter` to streamline the library and enhance focus on reading capabilities.
  - Fixed call method of `VideoReader`, now properly seeks frames.

- **New Features**:
  - Added `__getitem__` method to `VideoReader` for easier access to properties, allowing users to retrieve metadata using dictionary-like syntax (e.g., `reader['width']`).
  - Expanded `VideoReader.get_properties()` to include new metadata properties:
    - `codec`: **The name of the codec being used.**
    - `bit_depth`: **The bit-depth of the video.**
    - `has_audio`: **Indicates whether the video contains an audio track.**
    - `audio_bitrate`: **Bitrate of the audio stream.**
    - `audio_channels`: **Number of audio channels.**
    - `audio_sample_rate`: **Sample rate of the audio stream.**
    - `audio_codec`: **Codec used for audio.**
    - `min_fps`: **Minimum frames per second of the video.**
    - `max_fps`: **Maximum frames per second of the video.**
    - `aspect_ratio`: **Aspect ratio of the video.**
    
- **New Converter Formats**:
  - Completed the implementation of the following converters to support new video formats:
    - YUV420P to RGB
    - YUV420P10LE to RGB48
    - BGR to RGB
    - RGB to RGB
    - P010LE to RGB48

- **Supported Codecs**:
  - The following codecs can be worked with using the `VideoReader`, based on supported pixel formats:
    - **H.264 (AVC)**: YUV420P, YUV420P10LE
    - **H.265 (HEVC)**: YUV420P, YUV420P10LE
    - **VP8/VP9**: YUV420P, YUV420P10LE
    - **AV1**: YUV420P, YUV420P10LE
    - **MPEG-2**: YUV420P
    - **ProRes**: YUV420P, YUV422, YUV444
    - **DNxHD/DNxHR**: YUV422, YUV444
    - **DV (Digital Video)**: YUV420P
    - **Uncompressed RGB**: RGB, BGR
    - **P010LE**: P010LE

- **Testing Improvements**:
  - Updated tests to ensure compatibility with various bit-depths and codec types.
  - Added tests to verify the correct functionality of the new features and converters.


### Version 0.4.0 (2024-10-23)
  - Moved to `FFmpeg` static libraries!
    - Startup times are improved. All libs that can be static, are static. 
  - Adjusted logging to flow a little bit better, not overcrowd console unless desired. 
    - Logging details more info on codecs. The Decoder selects the **BEST** codec for the video.
  - Need to investigate if `NVDEC` is bottlenecked, or I've reached max performance capabilities. 
    - It is curious that cpu benches at `1859 fps` and gpu benches at `1809 fps`.

### Version 0.3.9 (2024-10-21)
 
- **Pre-Release Update:**
  - Prep for **0.4.0** release.
    - **0.4.x** release will be characterized by new codec and pixel format support!
    - Removed `d_type` and `buffer_size` arguments from `VideoReader` and `VideoWriter`.
      - Output and Input tensors are now, by standard, `UINT8`, `HWC` format, [0,255].
    - Standardized to `YUV420P` for now.
    - Swapped custom `CUDA` kernels for `nppi`. 
    - various cleanup and small refactorings.

### Version 0.3.8 (2024-10-21)
 
- **Pre-Release Update:**
  - Removed Buffering from `VideoWriter`, resulting in **INSANE** performance gains.
  - Fixed threading issue with `VideoWriter`, now properly utilizes available threads.
  - Removed `sync` method from `VideoWriter`. 
    - Synchronization can be manually handled by the user or by letting the `VideoWriter` do so on destruction. 
  - Updated Benchmarks to reflect new version.

### Version 0.3.7 (2024-10-21)

- **Pre-Release Update:**
  - Fixed remaining issues with `VideoWriter` class.
    - Both `cpu` and `cuda` arguments NOW work properly.
  - Few Small bug fixes regarding synchronization and memory management. 

### Version 0.3.6 (2024-10-19)

- **Pre-Release Update:**
  - Fixed `VideoWriter` class.
    - Both `cpu` and `cuda` arguments now work properly.
  - **Encoder Functionality:**
    - Enabled encoder support for both CPU and CUDA backends.
    - Users can now encode videos directly from PyTorch tensors.
  - Update Github Actions, add tests.

### Version 0.3.5 (2024-10-19)

- **Pre-Release Update:**
  - (somewhat) Fixed `VideoWriter` class. Working on `cuda` for now, but `cpu` still has incorrect output.
  - Added `VideoWriter`, and `LogLevel` definitions to `.pyi` stub file.
  - Adjusted github actions to publish to `pypi`.

### Version 0.3.4.1 (2024-10-19)

- **Pre-Release Update:**
  - Added logging utility for debugging purposes.
    ```py
    import celux
    celux.set_log_level(celux.LogLevel.debug)
    ```

### Version 0.3.3 (2024-10-19)

- **Pre-Release Update:**
  - Added `buffer_size` and `stream` arguments.
    - Choose Pre-Decoded Frame buffer size, and pass your own cuda stream.
  - Some random cleanup and small refactorings.

### Version 0.3.1 (2024-10-17)

- **Pre-Release Update:**
  - Adjusted Frame Range End in `VideoReader` to be exclusive to match `cv2` behavior.
  - Removed unnecessary error throws.
  - **Encoder Functionality:** Now fully operational for both CPU and CUDA.

### Version 0.3.0 (2024-10-17)

- **Pre-Release Update:**
  - Renamed from `ffmpy` to `CeLux`.
  - Created official `pypi` release.
  - Refactored to split `cpu` and `cuda` backends.

  
### Version 0.2.6 (2024-10-15)

- **Pre-Release Update:**
  - Removed `Numpy` support in favor of `PyTorch` tensors with GPU/CPU support.
  - Added `NV12ToBGR`, `BGRToNV12`, and `NV12ToNV12` conversion modules.
  - Fixed several minor issues.
  - Updated documentation and examples.

### Version 0.2.2 (2024-10-14)

- **Pre-Release Update:**
  - Fixed several minor issues.
  - Made `VideoReader` and `VideoWriter` callable.
  - Created BGR conversion modules.
  - Added frame range (in/out) arguments.

    ```python
    with VideoReader('input.mp4')([10, 20]) as reader:
        for frame in reader:
            print(f"Processing frame {frame}")
    ```

### Version 0.2.1 (2024-10-13)

- **Pre-Release Update:**
  - Adjusted Python bindings to use snake_case.
  - Added `.pyi` stub files to `.whl`.
  - Adjusted `dtype` arguments to (`uint8`, `float32`, `float16`).
  - Added GitHub Actions for new releases.
  - Added HW Accel Encoder support, direct encoding from numpy/tensors.
  - Added `has_audio` property to `VideoReader.get_properties()`.

### Version 0.1.1 (2024-10-06)

- **Pre-Release Update:**
  - Implemented support for multiple data types (`uint8`, `float`, `half`).
  - Provided example usage and basic documentation.