from concurrent.futures import thread
from typing import List, Optional, Any, Union, Tuple
import torch
from enum import Enum
import os

# For more info on FIlters, please visit https://ffmpeg.org/ffmpeg-filters.html

class LogLevel(Enum):
    trace = 0
    debug = 1
    info = 2
    warn = 3
    error = 4
    critical = 5
    off = 6
    
def set_log_level(level: LogLevel) -> None:
    """
    Set the logging level for CeLux.

    Args:
        level (LogLevel): The logging level to set.
    """
    ...

class VideoReader:
    def __init__(self, input_path: str, num_threads: int = os.cpu_count() // 2,
                 filters: Optional[List['FilterBase']] = None, tensor_shape: str = "HWC") -> None:
        """
        Initialize the VideoReader object.

        Args:
            input_path (str): Path to the video file.
            num_threads (int, optional): Number of threads for decoding. Defaults to half of CPU cores.
            filters (Optional[List[FilterBase]]): List of filters to apply to the video.
                - Example: filters = [("scale", "1280:720"), ("hue", "0.5")]
            tensor_shape (str, optional): Shape format of the output tensor. Defaults to "HWC".
        """
        ...

    def __call__(self, frame_range: Union[Tuple[int, int], Tuple[float, float],
                                          List[int], List[float]]) -> 'VideoReader':
        """
        Set the frame range for the video reader.

        This method allows setting a **range** using **frame indices (int)** or **timestamps (float)**.

        **Example:**
        ```python
        with VideoReader('input.mp4')([10, 20]) as reader:  # Using frames
            for frame in reader:
                process_frame(frame)

        with VideoReader('input.mp4')([1.5, 2.5]) as reader:  # Using timestamps
            for frame in reader:
                process_frame(frame)
        ```

        Args:
            frame_range (Union[Tuple[int, int], Tuple[float, float], List[int], List[float]]):
                A tuple or list containing the start and end frame indices or timestamps.

        Returns:
            VideoReader: The video reader object itself.

        Raises:
            ValueError: If `frame_range` is not a tuple or list of two **ints or two floats**.
        """
        ...

    def read_frame(self) -> torch.Tensor:
        """
        Read the next frame from the video.

        Returns:
            torch.Tensor: The frame data as a PyTorch tensor.
        """
        ...

    def seek(self, timestamp: float) -> bool:
        """
        Seek to a specific timestamp in the video.

        Args:
            timestamp (float): Timestamp in seconds.

        Returns:
            bool: True if the seek was successful, otherwise False.
        """
        ...

    def set_range(self, start: Union[int, float], end: Union[int, float]) -> None:
        """
        Set the playback range using either **frame numbers (int)** or **timestamps (float)**.

        **Example:**
        ```python
        reader.set_range(10, 50)      # Frame-based range
        reader.set_range(1.5, 2.5)    # Time-based range
        ```

        Args:
            start (Union[int, float]): Starting frame number or timestamp.
            end (Union[int, float]): Ending frame number or timestamp.

        Raises:
            ValueError: If `start` and `end` are not both **ints** or both **floats**.
        """
        ...

    def supported_codecs(self) -> List[str]:
        """
        Get a list of supported video codecs.

        Returns:
            List[str]: List of supported codec names.
        """
        ...

    def get_properties(self) -> dict:
        """
        Retrieve video properties.

        Returns:
            dict: A dictionary containing the following properties:
                - width (int): Video width.
                - height (int): Video height.
                - fps (float): Frames per second.
                - min_fps (float): Minimum frames per second.
                - max_fps (float): Maximum frames per second.
                - duration (float): Video duration in seconds.
                - total_frames (int): Total number of frames.
                - pixel_format (str): Video pixel format.
                - has_audio (bool): Whether the video contains audio.
                - audio_bitrate (Optional[int]): Audio bitrate (if applicable).
                - audio_channels (Optional[int]): Number of audio channels.
                - audio_sample_rate (Optional[int]): Audio sample rate.
                - audio_codec (Optional[str]): Audio codec name.
                - bit_depth (int): Bit depth of the video.
                - aspect_ratio (str): Video aspect ratio.
                - codec (str): Video codec name.
        """
        ...

    def __len__(self) -> int:
        """
        Get the total number of frames in the video.

        Returns:
            int: Number of frames in the video.
        """
        ...

    def __iter__(self) -> 'VideoReader':
        """
        Get an iterator for iterating over frames.

        Returns:
            VideoReader: The video reader object itself.
        """
        ...

    def __next__(self) -> torch.Tensor:
        """
        Retrieve the next frame in the video.

        Returns:
            torch.Tensor: The next frame as a PyTorch tensor.

        Raises:
            StopIteration: Raised when no more frames are available.
        """
        ...

    def __enter__(self) -> 'VideoReader':
        """
        Enter the context manager.

        Returns:
            VideoReader: The video reader object itself.
        """
        ...

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[BaseException], traceback: Optional[Any]) -> bool:
        """
        Exit the context manager.

        Args:
            exc_type (Optional[type]): The exception type, if any.
            exc_value (Optional[BaseException]): The exception value, if any.
            traceback (Optional[Any]): The traceback object, if any.

        Returns:
            bool: False to propagate exceptions, True to suppress them.
        """
        ...

    def reset(self) -> None:
        """
        Reset the video reader to the beginning of the video.
        """
        ...

    def __getitem__(self, key: str) -> Any:
        """
        Retrieve a specific property of the video.

        Args:
            key (str): Property name.

        Returns:
            Any: The value of the requested property.
        """
        ...

# This file is autogenerated. Do not modify manually.
from typing import List, Optional, Any, Tuple
from enum import Enum

class FilterBase:
    """
    Base class for all filters.
    """
    pass

class Acopy(FilterBase):
    """
    Copy the input audio unchanged to the output.
    """
    pass

class Aderivative(FilterBase):
    """
    Compute derivative of input audio.
    """
    pass

class Aintegral(FilterBase):
    """
    Compute integral of input audio.
    """
    pass

class Alatency(FilterBase):
    """
    Report audio filtering latency.
    """
    pass

class Amultiply(FilterBase):
    """
    Multiply two audio streams.
    """
    pass

class Anull(FilterBase):
    """
    Pass the source unchanged to the output.
    """
    pass

class Apsnr(FilterBase):
    """
    Measure Audio Peak Signal-to-Noise Ratio.
    """
    pass

class Areverse(FilterBase):
    """
    Reverse an audio clip.
    """
    pass

class Asdr(FilterBase):
    """
    Measure Audio Signal-to-Distortion Ratio.
    """
    pass

class Ashowinfo(FilterBase):
    """
    Show textual information for each audio frame.
    """
    pass

class Asisdr(FilterBase):
    """
    Measure Audio Scale-Invariant Signal-to-Distortion Ratio.
    """
    pass

class Earwax(FilterBase):
    """
    Widen the stereo image.
    """
    pass

class Volumedetect(FilterBase):
    """
    Detect audio volume.
    """
    pass

class Anullsink(FilterBase):
    """
    Do absolutely nothing with the input audio.
    """
    pass

class Addroi(FilterBase):
    """
    Add region of interest to frame.
    """
    def setRegionDistanceFromLeftEdgeOfFrame(self, value: Any) -> None:
        """
        Sets the value for regiondistancefromleftedgeofframe.

        Args:
            value (Any): The value to set.
        """
        ...
    def getRegionDistanceFromLeftEdgeOfFrame(self) -> Any:
        """
        Gets the value for regiondistancefromleftedgeofframe.

        Returns:
            Any: The current value.
        """
        ...
    def setRegionDistanceFromTopEdgeOfFrame(self, value: Any) -> None:
        """
        Sets the value for regiondistancefromtopedgeofframe.

        Args:
            value (Any): The value to set.
        """
        ...
    def getRegionDistanceFromTopEdgeOfFrame(self) -> Any:
        """
        Gets the value for regiondistancefromtopedgeofframe.

        Returns:
            Any: The current value.
        """
        ...
    def setRegionWidth(self, value: Any) -> None:
        """
        Sets the value for regionwidth.

        Args:
            value (Any): The value to set.
        """
        ...
    def getRegionWidth(self) -> Any:
        """
        Gets the value for regionwidth.

        Returns:
            Any: The current value.
        """
        ...
    def setRegionHeight(self, value: Any) -> None:
        """
        Sets the value for regionheight.

        Args:
            value (Any): The value to set.
        """
        ...
    def getRegionHeight(self) -> Any:
        """
        Gets the value for regionheight.

        Returns:
            Any: The current value.
        """
        ...
    def setQoffset(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for qoffset.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getQoffset(self) -> Tuple[int, int]:
        """
        Gets the value for qoffset.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setClear(self, value: bool) -> None:
        """
        Sets the value for clear.

        Args:
            value (bool): The value to set.
        """
        ...
    def getClear(self) -> bool:
        """
        Gets the value for clear.

        Returns:
            bool: The current value.
        """
        ...

class Alphaextract(FilterBase):
    """
    Extract an alpha channel as a grayscale image component.
    """
    pass

class Alphamerge(FilterBase):
    """
    Copy the luma value of the second input into the alpha channel of the first input.
    """
    pass

class Amplify(FilterBase):
    """
    Amplify changes between successive video frames.
    """
    def setRadius(self, value: int) -> None:
        """
        Sets the value for radius.

        Args:
            value (int): The value to set.
        """
        ...
    def getRadius(self) -> int:
        """
        Gets the value for radius.

        Returns:
            int: The current value.
        """
        ...
    def setFactor(self, value: float) -> None:
        """
        Sets the value for factor.

        Args:
            value (float): The value to set.
        """
        ...
    def getFactor(self) -> float:
        """
        Gets the value for factor.

        Returns:
            float: The current value.
        """
        ...
    def setThreshold(self, value: float) -> None:
        """
        Sets the value for threshold.

        Args:
            value (float): The value to set.
        """
        ...
    def getThreshold(self) -> float:
        """
        Gets the value for threshold.

        Returns:
            float: The current value.
        """
        ...
    def setTolerance(self, value: float) -> None:
        """
        Sets the value for tolerance.

        Args:
            value (float): The value to set.
        """
        ...
    def getTolerance(self) -> float:
        """
        Gets the value for tolerance.

        Returns:
            float: The current value.
        """
        ...
    def setLow(self, value: float) -> None:
        """
        Sets the value for low.

        Args:
            value (float): The value to set.
        """
        ...
    def getLow(self) -> float:
        """
        Gets the value for low.

        Returns:
            float: The current value.
        """
        ...
    def setHigh(self, value: float) -> None:
        """
        Sets the value for high.

        Args:
            value (float): The value to set.
        """
        ...
    def getHigh(self) -> float:
        """
        Gets the value for high.

        Returns:
            float: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Atadenoise(FilterBase):
    """
    Apply an Adaptive Temporal Averaging Denoiser.
    """
    def set_0a(self, value: float) -> None:
        """
        Sets the value for _0a.

        Args:
            value (float): The value to set.
        """
        ...
    def get_0a(self) -> float:
        """
        Gets the value for _0a.

        Returns:
            float: The current value.
        """
        ...
    def set_0b(self, value: float) -> None:
        """
        Sets the value for _0b.

        Args:
            value (float): The value to set.
        """
        ...
    def get_0b(self) -> float:
        """
        Gets the value for _0b.

        Returns:
            float: The current value.
        """
        ...
    def set_1a(self, value: float) -> None:
        """
        Sets the value for _1a.

        Args:
            value (float): The value to set.
        """
        ...
    def get_1a(self) -> float:
        """
        Gets the value for _1a.

        Returns:
            float: The current value.
        """
        ...
    def set_1b(self, value: float) -> None:
        """
        Sets the value for _1b.

        Args:
            value (float): The value to set.
        """
        ...
    def get_1b(self) -> float:
        """
        Gets the value for _1b.

        Returns:
            float: The current value.
        """
        ...
    def set_2a(self, value: float) -> None:
        """
        Sets the value for _2a.

        Args:
            value (float): The value to set.
        """
        ...
    def get_2a(self) -> float:
        """
        Gets the value for _2a.

        Returns:
            float: The current value.
        """
        ...
    def set_2b(self, value: float) -> None:
        """
        Sets the value for _2b.

        Args:
            value (float): The value to set.
        """
        ...
    def get_2b(self) -> float:
        """
        Gets the value for _2b.

        Returns:
            float: The current value.
        """
        ...
    def setHowManyFramesToUse(self, value: int) -> None:
        """
        Sets the value for howmanyframestouse.

        Args:
            value (int): The value to set.
        """
        ...
    def getHowManyFramesToUse(self) -> int:
        """
        Gets the value for howmanyframestouse.

        Returns:
            int: The current value.
        """
        ...
    def setWhatPlanesToFilter(self, value: int) -> None:
        """
        Sets the value for whatplanestofilter.

        Args:
            value (int): The value to set.
        """
        ...
    def getWhatPlanesToFilter(self) -> int:
        """
        Gets the value for whatplanestofilter.

        Returns:
            int: The current value.
        """
        ...
    def setVariantOfAlgorithm(self, value: int) -> None:
        """
        Sets the value for variantofalgorithm.

        Args:
            value (int): The value to set.
        """
        ...
    def getVariantOfAlgorithm(self) -> int:
        """
        Gets the value for variantofalgorithm.

        Returns:
            int: The current value.
        """
        ...
    def set_0s(self, value: float) -> None:
        """
        Sets the value for _0s.

        Args:
            value (float): The value to set.
        """
        ...
    def get_0s(self) -> float:
        """
        Gets the value for _0s.

        Returns:
            float: The current value.
        """
        ...
    def set_1s(self, value: float) -> None:
        """
        Sets the value for _1s.

        Args:
            value (float): The value to set.
        """
        ...
    def get_1s(self) -> float:
        """
        Gets the value for _1s.

        Returns:
            float: The current value.
        """
        ...
    def set_2s(self, value: float) -> None:
        """
        Sets the value for _2s.

        Args:
            value (float): The value to set.
        """
        ...
    def get_2s(self) -> float:
        """
        Gets the value for _2s.

        Returns:
            float: The current value.
        """
        ...

class Avgblur(FilterBase):
    """
    Apply Average Blur filter.
    """
    def setSizeX(self, value: int) -> None:
        """
        Sets the value for sizex.

        Args:
            value (int): The value to set.
        """
        ...
    def getSizeX(self) -> int:
        """
        Gets the value for sizex.

        Returns:
            int: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setSizeY(self, value: int) -> None:
        """
        Sets the value for sizey.

        Args:
            value (int): The value to set.
        """
        ...
    def getSizeY(self) -> int:
        """
        Gets the value for sizey.

        Returns:
            int: The current value.
        """
        ...

class Backgroundkey(FilterBase):
    """
    Turns a static background into transparency.
    """
    def setThreshold(self, value: float) -> None:
        """
        Sets the value for threshold.

        Args:
            value (float): The value to set.
        """
        ...
    def getThreshold(self) -> float:
        """
        Gets the value for threshold.

        Returns:
            float: The current value.
        """
        ...
    def setSimilarity(self, value: float) -> None:
        """
        Sets the value for similarity.

        Args:
            value (float): The value to set.
        """
        ...
    def getSimilarity(self) -> float:
        """
        Gets the value for similarity.

        Returns:
            float: The current value.
        """
        ...
    def setBlend(self, value: float) -> None:
        """
        Sets the value for blend.

        Args:
            value (float): The value to set.
        """
        ...
    def getBlend(self) -> float:
        """
        Gets the value for blend.

        Returns:
            float: The current value.
        """
        ...

class Bbox(FilterBase):
    """
    Compute bounding box for each frame.
    """
    def setMin_val(self, value: int) -> None:
        """
        Sets the value for min_val.

        Args:
            value (int): The value to set.
        """
        ...
    def getMin_val(self) -> int:
        """
        Gets the value for min_val.

        Returns:
            int: The current value.
        """
        ...

class Bench(FilterBase):
    """
    Benchmark part of a filtergraph.
    """
    def setAction(self, value: int) -> None:
        """
        Sets the value for action.

        Args:
            value (int): The value to set.
        """
        ...
    def getAction(self) -> int:
        """
        Gets the value for action.

        Returns:
            int: The current value.
        """
        ...

class Bilateral(FilterBase):
    """
    Apply Bilateral filter.
    """
    def setSigmaS(self, value: float) -> None:
        """
        Sets the value for sigmas.

        Args:
            value (float): The value to set.
        """
        ...
    def getSigmaS(self) -> float:
        """
        Gets the value for sigmas.

        Returns:
            float: The current value.
        """
        ...
    def setSigmaR(self, value: float) -> None:
        """
        Sets the value for sigmar.

        Args:
            value (float): The value to set.
        """
        ...
    def getSigmaR(self) -> float:
        """
        Gets the value for sigmar.

        Returns:
            float: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Bitplanenoise(FilterBase):
    """
    Measure bit plane noise.
    """
    def setBitplane(self, value: int) -> None:
        """
        Sets the value for bitplane.

        Args:
            value (int): The value to set.
        """
        ...
    def getBitplane(self) -> int:
        """
        Gets the value for bitplane.

        Returns:
            int: The current value.
        """
        ...
    def setFilter(self, value: bool) -> None:
        """
        Sets the value for filter.

        Args:
            value (bool): The value to set.
        """
        ...
    def getFilter(self) -> bool:
        """
        Gets the value for filter.

        Returns:
            bool: The current value.
        """
        ...

class Blackdetect(FilterBase):
    """
    Detect video intervals that are (almost) black.
    """
    def setBlack_min_duration(self, value: float) -> None:
        """
        Sets the value for black_min_duration.

        Args:
            value (float): The value to set.
        """
        ...
    def getBlack_min_duration(self) -> float:
        """
        Gets the value for black_min_duration.

        Returns:
            float: The current value.
        """
        ...
    def setPicture_black_ratio_th(self, value: float) -> None:
        """
        Sets the value for picture_black_ratio_th.

        Args:
            value (float): The value to set.
        """
        ...
    def getPicture_black_ratio_th(self) -> float:
        """
        Gets the value for picture_black_ratio_th.

        Returns:
            float: The current value.
        """
        ...
    def setPixel_black_th(self, value: float) -> None:
        """
        Sets the value for pixel_black_th.

        Args:
            value (float): The value to set.
        """
        ...
    def getPixel_black_th(self) -> float:
        """
        Gets the value for pixel_black_th.

        Returns:
            float: The current value.
        """
        ...

class Blend(FilterBase):
    """
    Blend two video frames into each other.
    """
    def setC0_mode(self, value: int) -> None:
        """
        Sets the value for c0_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getC0_mode(self) -> int:
        """
        Gets the value for c0_mode.

        Returns:
            int: The current value.
        """
        ...
    def setC1_mode(self, value: int) -> None:
        """
        Sets the value for c1_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getC1_mode(self) -> int:
        """
        Gets the value for c1_mode.

        Returns:
            int: The current value.
        """
        ...
    def setC2_mode(self, value: int) -> None:
        """
        Sets the value for c2_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getC2_mode(self) -> int:
        """
        Gets the value for c2_mode.

        Returns:
            int: The current value.
        """
        ...
    def setC3_mode(self, value: int) -> None:
        """
        Sets the value for c3_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getC3_mode(self) -> int:
        """
        Gets the value for c3_mode.

        Returns:
            int: The current value.
        """
        ...
    def setAll_mode(self, value: int) -> None:
        """
        Sets the value for all_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getAll_mode(self) -> int:
        """
        Gets the value for all_mode.

        Returns:
            int: The current value.
        """
        ...
    def setC0_expr(self, value: Any) -> None:
        """
        Sets the value for c0_expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC0_expr(self) -> Any:
        """
        Gets the value for c0_expr.

        Returns:
            Any: The current value.
        """
        ...
    def setC1_expr(self, value: Any) -> None:
        """
        Sets the value for c1_expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC1_expr(self) -> Any:
        """
        Gets the value for c1_expr.

        Returns:
            Any: The current value.
        """
        ...
    def setC2_expr(self, value: Any) -> None:
        """
        Sets the value for c2_expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC2_expr(self) -> Any:
        """
        Gets the value for c2_expr.

        Returns:
            Any: The current value.
        """
        ...
    def setC3_expr(self, value: Any) -> None:
        """
        Sets the value for c3_expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC3_expr(self) -> Any:
        """
        Gets the value for c3_expr.

        Returns:
            Any: The current value.
        """
        ...
    def setAll_expr(self, value: Any) -> None:
        """
        Sets the value for all_expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getAll_expr(self) -> Any:
        """
        Gets the value for all_expr.

        Returns:
            Any: The current value.
        """
        ...
    def setC0_opacity(self, value: float) -> None:
        """
        Sets the value for c0_opacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getC0_opacity(self) -> float:
        """
        Gets the value for c0_opacity.

        Returns:
            float: The current value.
        """
        ...
    def setC1_opacity(self, value: float) -> None:
        """
        Sets the value for c1_opacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getC1_opacity(self) -> float:
        """
        Gets the value for c1_opacity.

        Returns:
            float: The current value.
        """
        ...
    def setC2_opacity(self, value: float) -> None:
        """
        Sets the value for c2_opacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getC2_opacity(self) -> float:
        """
        Gets the value for c2_opacity.

        Returns:
            float: The current value.
        """
        ...
    def setC3_opacity(self, value: float) -> None:
        """
        Sets the value for c3_opacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getC3_opacity(self) -> float:
        """
        Gets the value for c3_opacity.

        Returns:
            float: The current value.
        """
        ...
    def setAll_opacity(self, value: float) -> None:
        """
        Sets the value for all_opacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getAll_opacity(self) -> float:
        """
        Gets the value for all_opacity.

        Returns:
            float: The current value.
        """
        ...

class Blockdetect(FilterBase):
    """
    Blockdetect filter.
    """
    def setPeriod_min(self, value: int) -> None:
        """
        Sets the value for period_min.

        Args:
            value (int): The value to set.
        """
        ...
    def getPeriod_min(self) -> int:
        """
        Gets the value for period_min.

        Returns:
            int: The current value.
        """
        ...
    def setPeriod_max(self, value: int) -> None:
        """
        Sets the value for period_max.

        Args:
            value (int): The value to set.
        """
        ...
    def getPeriod_max(self) -> int:
        """
        Gets the value for period_max.

        Returns:
            int: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Blurdetect(FilterBase):
    """
    Blurdetect filter.
    """
    def setHigh(self, value: float) -> None:
        """
        Sets the value for high.

        Args:
            value (float): The value to set.
        """
        ...
    def getHigh(self) -> float:
        """
        Gets the value for high.

        Returns:
            float: The current value.
        """
        ...
    def setLow(self, value: float) -> None:
        """
        Sets the value for low.

        Args:
            value (float): The value to set.
        """
        ...
    def getLow(self) -> float:
        """
        Gets the value for low.

        Returns:
            float: The current value.
        """
        ...
    def setRadius(self, value: int) -> None:
        """
        Sets the value for radius.

        Args:
            value (int): The value to set.
        """
        ...
    def getRadius(self) -> int:
        """
        Gets the value for radius.

        Returns:
            int: The current value.
        """
        ...
    def setBlock_pct(self, value: int) -> None:
        """
        Sets the value for block_pct.

        Args:
            value (int): The value to set.
        """
        ...
    def getBlock_pct(self) -> int:
        """
        Gets the value for block_pct.

        Returns:
            int: The current value.
        """
        ...
    def setBlock_height(self, value: int) -> None:
        """
        Sets the value for block_height.

        Args:
            value (int): The value to set.
        """
        ...
    def getBlock_height(self) -> int:
        """
        Gets the value for block_height.

        Returns:
            int: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Bm3d(FilterBase):
    """
    Block-Matching 3D denoiser.
    """
    def setSigma(self, value: float) -> None:
        """
        Sets the value for sigma.

        Args:
            value (float): The value to set.
        """
        ...
    def getSigma(self) -> float:
        """
        Gets the value for sigma.

        Returns:
            float: The current value.
        """
        ...
    def setBlock(self, value: int) -> None:
        """
        Sets the value for block.

        Args:
            value (int): The value to set.
        """
        ...
    def getBlock(self) -> int:
        """
        Gets the value for block.

        Returns:
            int: The current value.
        """
        ...
    def setBstep(self, value: int) -> None:
        """
        Sets the value for bstep.

        Args:
            value (int): The value to set.
        """
        ...
    def getBstep(self) -> int:
        """
        Gets the value for bstep.

        Returns:
            int: The current value.
        """
        ...
    def setGroup(self, value: int) -> None:
        """
        Sets the value for group.

        Args:
            value (int): The value to set.
        """
        ...
    def getGroup(self) -> int:
        """
        Gets the value for group.

        Returns:
            int: The current value.
        """
        ...
    def setRange(self, value: int) -> None:
        """
        Sets the value for range.

        Args:
            value (int): The value to set.
        """
        ...
    def getRange(self) -> int:
        """
        Gets the value for range.

        Returns:
            int: The current value.
        """
        ...
    def setMstep(self, value: int) -> None:
        """
        Sets the value for mstep.

        Args:
            value (int): The value to set.
        """
        ...
    def getMstep(self) -> int:
        """
        Gets the value for mstep.

        Returns:
            int: The current value.
        """
        ...
    def setThmse(self, value: float) -> None:
        """
        Sets the value for thmse.

        Args:
            value (float): The value to set.
        """
        ...
    def getThmse(self) -> float:
        """
        Gets the value for thmse.

        Returns:
            float: The current value.
        """
        ...
    def setHdthr(self, value: float) -> None:
        """
        Sets the value for hdthr.

        Args:
            value (float): The value to set.
        """
        ...
    def getHdthr(self) -> float:
        """
        Gets the value for hdthr.

        Returns:
            float: The current value.
        """
        ...
    def setEstim(self, value: int) -> None:
        """
        Sets the value for estim.

        Args:
            value (int): The value to set.
        """
        ...
    def getEstim(self) -> int:
        """
        Gets the value for estim.

        Returns:
            int: The current value.
        """
        ...
    def setRef(self, value: bool) -> None:
        """
        Sets the value for ref.

        Args:
            value (bool): The value to set.
        """
        ...
    def getRef(self) -> bool:
        """
        Gets the value for ref.

        Returns:
            bool: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Bwdif(FilterBase):
    """
    Deinterlace the input image.
    """
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setParity(self, value: int) -> None:
        """
        Sets the value for parity.

        Args:
            value (int): The value to set.
        """
        ...
    def getParity(self) -> int:
        """
        Gets the value for parity.

        Returns:
            int: The current value.
        """
        ...
    def setDeint(self, value: int) -> None:
        """
        Sets the value for deint.

        Args:
            value (int): The value to set.
        """
        ...
    def getDeint(self) -> int:
        """
        Gets the value for deint.

        Returns:
            int: The current value.
        """
        ...

class Cas(FilterBase):
    """
    Contrast Adaptive Sharpen.
    """
    def setStrength(self, value: float) -> None:
        """
        Sets the value for strength.

        Args:
            value (float): The value to set.
        """
        ...
    def getStrength(self) -> float:
        """
        Gets the value for strength.

        Returns:
            float: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Ccrepack(FilterBase):
    """
    Repack CEA-708 closed caption metadata
    """
    pass

class Chromahold(FilterBase):
    """
    Turns a certain color range into gray.
    """
    def setColor(self, value: Any) -> None:
        """
        Sets the value for color.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColor(self) -> Any:
        """
        Gets the value for color.

        Returns:
            Any: The current value.
        """
        ...
    def setSimilarity(self, value: float) -> None:
        """
        Sets the value for similarity.

        Args:
            value (float): The value to set.
        """
        ...
    def getSimilarity(self) -> float:
        """
        Gets the value for similarity.

        Returns:
            float: The current value.
        """
        ...
    def setBlend(self, value: float) -> None:
        """
        Sets the value for blend.

        Args:
            value (float): The value to set.
        """
        ...
    def getBlend(self) -> float:
        """
        Gets the value for blend.

        Returns:
            float: The current value.
        """
        ...
    def setYuv(self, value: bool) -> None:
        """
        Sets the value for yuv.

        Args:
            value (bool): The value to set.
        """
        ...
    def getYuv(self) -> bool:
        """
        Gets the value for yuv.

        Returns:
            bool: The current value.
        """
        ...

class Chromakey(FilterBase):
    """
    Turns a certain color into transparency. Operates on YUV colors.
    """
    def setColor(self, value: Any) -> None:
        """
        Sets the value for color.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColor(self) -> Any:
        """
        Gets the value for color.

        Returns:
            Any: The current value.
        """
        ...
    def setSimilarity(self, value: float) -> None:
        """
        Sets the value for similarity.

        Args:
            value (float): The value to set.
        """
        ...
    def getSimilarity(self) -> float:
        """
        Gets the value for similarity.

        Returns:
            float: The current value.
        """
        ...
    def setBlend(self, value: float) -> None:
        """
        Sets the value for blend.

        Args:
            value (float): The value to set.
        """
        ...
    def getBlend(self) -> float:
        """
        Gets the value for blend.

        Returns:
            float: The current value.
        """
        ...
    def setYuv(self, value: bool) -> None:
        """
        Sets the value for yuv.

        Args:
            value (bool): The value to set.
        """
        ...
    def getYuv(self) -> bool:
        """
        Gets the value for yuv.

        Returns:
            bool: The current value.
        """
        ...

class Chromanr(FilterBase):
    """
    Reduce chrominance noise.
    """
    def setThres(self, value: float) -> None:
        """
        Sets the value for thres.

        Args:
            value (float): The value to set.
        """
        ...
    def getThres(self) -> float:
        """
        Gets the value for thres.

        Returns:
            float: The current value.
        """
        ...
    def setSizew(self, value: int) -> None:
        """
        Sets the value for sizew.

        Args:
            value (int): The value to set.
        """
        ...
    def getSizew(self) -> int:
        """
        Gets the value for sizew.

        Returns:
            int: The current value.
        """
        ...
    def setSizeh(self, value: int) -> None:
        """
        Sets the value for sizeh.

        Args:
            value (int): The value to set.
        """
        ...
    def getSizeh(self) -> int:
        """
        Gets the value for sizeh.

        Returns:
            int: The current value.
        """
        ...
    def setStepw(self, value: int) -> None:
        """
        Sets the value for stepw.

        Args:
            value (int): The value to set.
        """
        ...
    def getStepw(self) -> int:
        """
        Gets the value for stepw.

        Returns:
            int: The current value.
        """
        ...
    def setSteph(self, value: int) -> None:
        """
        Sets the value for steph.

        Args:
            value (int): The value to set.
        """
        ...
    def getSteph(self) -> int:
        """
        Gets the value for steph.

        Returns:
            int: The current value.
        """
        ...
    def setThrey(self, value: float) -> None:
        """
        Sets the value for threy.

        Args:
            value (float): The value to set.
        """
        ...
    def getThrey(self) -> float:
        """
        Gets the value for threy.

        Returns:
            float: The current value.
        """
        ...
    def setThreu(self, value: float) -> None:
        """
        Sets the value for threu.

        Args:
            value (float): The value to set.
        """
        ...
    def getThreu(self) -> float:
        """
        Gets the value for threu.

        Returns:
            float: The current value.
        """
        ...
    def setThrev(self, value: float) -> None:
        """
        Sets the value for threv.

        Args:
            value (float): The value to set.
        """
        ...
    def getThrev(self) -> float:
        """
        Gets the value for threv.

        Returns:
            float: The current value.
        """
        ...
    def setDistance(self, value: int) -> None:
        """
        Sets the value for distance.

        Args:
            value (int): The value to set.
        """
        ...
    def getDistance(self) -> int:
        """
        Gets the value for distance.

        Returns:
            int: The current value.
        """
        ...

class Chromashift(FilterBase):
    """
    Shift chroma.
    """
    def setCbh(self, value: int) -> None:
        """
        Sets the value for cbh.

        Args:
            value (int): The value to set.
        """
        ...
    def getCbh(self) -> int:
        """
        Gets the value for cbh.

        Returns:
            int: The current value.
        """
        ...
    def setCbv(self, value: int) -> None:
        """
        Sets the value for cbv.

        Args:
            value (int): The value to set.
        """
        ...
    def getCbv(self) -> int:
        """
        Gets the value for cbv.

        Returns:
            int: The current value.
        """
        ...
    def setCrh(self, value: int) -> None:
        """
        Sets the value for crh.

        Args:
            value (int): The value to set.
        """
        ...
    def getCrh(self) -> int:
        """
        Gets the value for crh.

        Returns:
            int: The current value.
        """
        ...
    def setCrv(self, value: int) -> None:
        """
        Sets the value for crv.

        Args:
            value (int): The value to set.
        """
        ...
    def getCrv(self) -> int:
        """
        Gets the value for crv.

        Returns:
            int: The current value.
        """
        ...
    def setEdge(self, value: int) -> None:
        """
        Sets the value for edge.

        Args:
            value (int): The value to set.
        """
        ...
    def getEdge(self) -> int:
        """
        Gets the value for edge.

        Returns:
            int: The current value.
        """
        ...

class Ciescope(FilterBase):
    """
    Video CIE scope.
    """
    def setSystem(self, value: int) -> None:
        """
        Sets the value for system.

        Args:
            value (int): The value to set.
        """
        ...
    def getSystem(self) -> int:
        """
        Gets the value for system.

        Returns:
            int: The current value.
        """
        ...
    def setCie(self, value: int) -> None:
        """
        Sets the value for cie.

        Args:
            value (int): The value to set.
        """
        ...
    def getCie(self) -> int:
        """
        Gets the value for cie.

        Returns:
            int: The current value.
        """
        ...
    def setGamuts(self, value: int) -> None:
        """
        Sets the value for gamuts.

        Args:
            value (int): The value to set.
        """
        ...
    def getGamuts(self) -> int:
        """
        Gets the value for gamuts.

        Returns:
            int: The current value.
        """
        ...
    def setSize(self, value: int) -> None:
        """
        Sets the value for size.

        Args:
            value (int): The value to set.
        """
        ...
    def getSize(self) -> int:
        """
        Gets the value for size.

        Returns:
            int: The current value.
        """
        ...
    def setIntensity(self, value: float) -> None:
        """
        Sets the value for intensity.

        Args:
            value (float): The value to set.
        """
        ...
    def getIntensity(self) -> float:
        """
        Gets the value for intensity.

        Returns:
            float: The current value.
        """
        ...
    def setContrast(self, value: float) -> None:
        """
        Sets the value for contrast.

        Args:
            value (float): The value to set.
        """
        ...
    def getContrast(self) -> float:
        """
        Gets the value for contrast.

        Returns:
            float: The current value.
        """
        ...
    def setCorrgamma(self, value: bool) -> None:
        """
        Sets the value for corrgamma.

        Args:
            value (bool): The value to set.
        """
        ...
    def getCorrgamma(self) -> bool:
        """
        Gets the value for corrgamma.

        Returns:
            bool: The current value.
        """
        ...
    def setShowwhite(self, value: bool) -> None:
        """
        Sets the value for showwhite.

        Args:
            value (bool): The value to set.
        """
        ...
    def getShowwhite(self) -> bool:
        """
        Gets the value for showwhite.

        Returns:
            bool: The current value.
        """
        ...
    def setGamma(self, value: float) -> None:
        """
        Sets the value for gamma.

        Args:
            value (float): The value to set.
        """
        ...
    def getGamma(self) -> float:
        """
        Gets the value for gamma.

        Returns:
            float: The current value.
        """
        ...
    def setFill(self, value: bool) -> None:
        """
        Sets the value for fill.

        Args:
            value (bool): The value to set.
        """
        ...
    def getFill(self) -> bool:
        """
        Gets the value for fill.

        Returns:
            bool: The current value.
        """
        ...

class Codecview(FilterBase):
    """
    Visualize information about some codecs.
    """
    def setMv(self, value: int) -> None:
        """
        Sets the value for mv.

        Args:
            value (int): The value to set.
        """
        ...
    def getMv(self) -> int:
        """
        Gets the value for mv.

        Returns:
            int: The current value.
        """
        ...
    def setQp(self, value: bool) -> None:
        """
        Sets the value for qp.

        Args:
            value (bool): The value to set.
        """
        ...
    def getQp(self) -> bool:
        """
        Gets the value for qp.

        Returns:
            bool: The current value.
        """
        ...
    def setMv_type(self, value: int) -> None:
        """
        Sets the value for mv_type.

        Args:
            value (int): The value to set.
        """
        ...
    def getMv_type(self) -> int:
        """
        Gets the value for mv_type.

        Returns:
            int: The current value.
        """
        ...
    def setFrame_type(self, value: int) -> None:
        """
        Sets the value for frame_type.

        Args:
            value (int): The value to set.
        """
        ...
    def getFrame_type(self) -> int:
        """
        Gets the value for frame_type.

        Returns:
            int: The current value.
        """
        ...
    def setBlock(self, value: bool) -> None:
        """
        Sets the value for block.

        Args:
            value (bool): The value to set.
        """
        ...
    def getBlock(self) -> bool:
        """
        Gets the value for block.

        Returns:
            bool: The current value.
        """
        ...

class Colorbalance(FilterBase):
    """
    Adjust the color balance.
    """
    def setRs(self, value: float) -> None:
        """
        Sets the value for rs.

        Args:
            value (float): The value to set.
        """
        ...
    def getRs(self) -> float:
        """
        Gets the value for rs.

        Returns:
            float: The current value.
        """
        ...
    def setGs(self, value: float) -> None:
        """
        Sets the value for gs.

        Args:
            value (float): The value to set.
        """
        ...
    def getGs(self) -> float:
        """
        Gets the value for gs.

        Returns:
            float: The current value.
        """
        ...
    def setBs(self, value: float) -> None:
        """
        Sets the value for bs.

        Args:
            value (float): The value to set.
        """
        ...
    def getBs(self) -> float:
        """
        Gets the value for bs.

        Returns:
            float: The current value.
        """
        ...
    def setRm(self, value: float) -> None:
        """
        Sets the value for rm.

        Args:
            value (float): The value to set.
        """
        ...
    def getRm(self) -> float:
        """
        Gets the value for rm.

        Returns:
            float: The current value.
        """
        ...
    def setGm(self, value: float) -> None:
        """
        Sets the value for gm.

        Args:
            value (float): The value to set.
        """
        ...
    def getGm(self) -> float:
        """
        Gets the value for gm.

        Returns:
            float: The current value.
        """
        ...
    def setBm(self, value: float) -> None:
        """
        Sets the value for bm.

        Args:
            value (float): The value to set.
        """
        ...
    def getBm(self) -> float:
        """
        Gets the value for bm.

        Returns:
            float: The current value.
        """
        ...
    def setRh(self, value: float) -> None:
        """
        Sets the value for rh.

        Args:
            value (float): The value to set.
        """
        ...
    def getRh(self) -> float:
        """
        Gets the value for rh.

        Returns:
            float: The current value.
        """
        ...
    def setGh(self, value: float) -> None:
        """
        Sets the value for gh.

        Args:
            value (float): The value to set.
        """
        ...
    def getGh(self) -> float:
        """
        Gets the value for gh.

        Returns:
            float: The current value.
        """
        ...
    def setBh(self, value: float) -> None:
        """
        Sets the value for bh.

        Args:
            value (float): The value to set.
        """
        ...
    def getBh(self) -> float:
        """
        Gets the value for bh.

        Returns:
            float: The current value.
        """
        ...
    def setPl(self, value: bool) -> None:
        """
        Sets the value for pl.

        Args:
            value (bool): The value to set.
        """
        ...
    def getPl(self) -> bool:
        """
        Gets the value for pl.

        Returns:
            bool: The current value.
        """
        ...

class Colorchannelmixer(FilterBase):
    """
    Adjust colors by mixing color channels.
    """
    def setRr(self, value: float) -> None:
        """
        Sets the value for rr.

        Args:
            value (float): The value to set.
        """
        ...
    def getRr(self) -> float:
        """
        Gets the value for rr.

        Returns:
            float: The current value.
        """
        ...
    def setRg(self, value: float) -> None:
        """
        Sets the value for rg.

        Args:
            value (float): The value to set.
        """
        ...
    def getRg(self) -> float:
        """
        Gets the value for rg.

        Returns:
            float: The current value.
        """
        ...
    def setRb(self, value: float) -> None:
        """
        Sets the value for rb.

        Args:
            value (float): The value to set.
        """
        ...
    def getRb(self) -> float:
        """
        Gets the value for rb.

        Returns:
            float: The current value.
        """
        ...
    def setRa(self, value: float) -> None:
        """
        Sets the value for ra.

        Args:
            value (float): The value to set.
        """
        ...
    def getRa(self) -> float:
        """
        Gets the value for ra.

        Returns:
            float: The current value.
        """
        ...
    def setGr(self, value: float) -> None:
        """
        Sets the value for gr.

        Args:
            value (float): The value to set.
        """
        ...
    def getGr(self) -> float:
        """
        Gets the value for gr.

        Returns:
            float: The current value.
        """
        ...
    def setGg(self, value: float) -> None:
        """
        Sets the value for gg.

        Args:
            value (float): The value to set.
        """
        ...
    def getGg(self) -> float:
        """
        Gets the value for gg.

        Returns:
            float: The current value.
        """
        ...
    def setGb(self, value: float) -> None:
        """
        Sets the value for gb.

        Args:
            value (float): The value to set.
        """
        ...
    def getGb(self) -> float:
        """
        Gets the value for gb.

        Returns:
            float: The current value.
        """
        ...
    def setGa(self, value: float) -> None:
        """
        Sets the value for ga.

        Args:
            value (float): The value to set.
        """
        ...
    def getGa(self) -> float:
        """
        Gets the value for ga.

        Returns:
            float: The current value.
        """
        ...
    def setBr(self, value: float) -> None:
        """
        Sets the value for br.

        Args:
            value (float): The value to set.
        """
        ...
    def getBr(self) -> float:
        """
        Gets the value for br.

        Returns:
            float: The current value.
        """
        ...
    def setBg(self, value: float) -> None:
        """
        Sets the value for bg.

        Args:
            value (float): The value to set.
        """
        ...
    def getBg(self) -> float:
        """
        Gets the value for bg.

        Returns:
            float: The current value.
        """
        ...
    def setBb(self, value: float) -> None:
        """
        Sets the value for bb.

        Args:
            value (float): The value to set.
        """
        ...
    def getBb(self) -> float:
        """
        Gets the value for bb.

        Returns:
            float: The current value.
        """
        ...
    def setBa(self, value: float) -> None:
        """
        Sets the value for ba.

        Args:
            value (float): The value to set.
        """
        ...
    def getBa(self) -> float:
        """
        Gets the value for ba.

        Returns:
            float: The current value.
        """
        ...
    def setAr(self, value: float) -> None:
        """
        Sets the value for ar.

        Args:
            value (float): The value to set.
        """
        ...
    def getAr(self) -> float:
        """
        Gets the value for ar.

        Returns:
            float: The current value.
        """
        ...
    def setAg(self, value: float) -> None:
        """
        Sets the value for ag.

        Args:
            value (float): The value to set.
        """
        ...
    def getAg(self) -> float:
        """
        Gets the value for ag.

        Returns:
            float: The current value.
        """
        ...
    def setAb(self, value: float) -> None:
        """
        Sets the value for ab.

        Args:
            value (float): The value to set.
        """
        ...
    def getAb(self) -> float:
        """
        Gets the value for ab.

        Returns:
            float: The current value.
        """
        ...
    def setAa(self, value: float) -> None:
        """
        Sets the value for aa.

        Args:
            value (float): The value to set.
        """
        ...
    def getAa(self) -> float:
        """
        Gets the value for aa.

        Returns:
            float: The current value.
        """
        ...
    def setPc(self, value: int) -> None:
        """
        Sets the value for pc.

        Args:
            value (int): The value to set.
        """
        ...
    def getPc(self) -> int:
        """
        Gets the value for pc.

        Returns:
            int: The current value.
        """
        ...
    def setPa(self, value: float) -> None:
        """
        Sets the value for pa.

        Args:
            value (float): The value to set.
        """
        ...
    def getPa(self) -> float:
        """
        Gets the value for pa.

        Returns:
            float: The current value.
        """
        ...

class Colorcontrast(FilterBase):
    """
    Adjust color contrast between RGB components.
    """
    def setRc(self, value: float) -> None:
        """
        Sets the value for rc.

        Args:
            value (float): The value to set.
        """
        ...
    def getRc(self) -> float:
        """
        Gets the value for rc.

        Returns:
            float: The current value.
        """
        ...
    def setGm(self, value: float) -> None:
        """
        Sets the value for gm.

        Args:
            value (float): The value to set.
        """
        ...
    def getGm(self) -> float:
        """
        Gets the value for gm.

        Returns:
            float: The current value.
        """
        ...
    def setBy(self, value: float) -> None:
        """
        Sets the value for by.

        Args:
            value (float): The value to set.
        """
        ...
    def getBy(self) -> float:
        """
        Gets the value for by.

        Returns:
            float: The current value.
        """
        ...
    def setRcw(self, value: float) -> None:
        """
        Sets the value for rcw.

        Args:
            value (float): The value to set.
        """
        ...
    def getRcw(self) -> float:
        """
        Gets the value for rcw.

        Returns:
            float: The current value.
        """
        ...
    def setGmw(self, value: float) -> None:
        """
        Sets the value for gmw.

        Args:
            value (float): The value to set.
        """
        ...
    def getGmw(self) -> float:
        """
        Gets the value for gmw.

        Returns:
            float: The current value.
        """
        ...
    def setByw(self, value: float) -> None:
        """
        Sets the value for byw.

        Args:
            value (float): The value to set.
        """
        ...
    def getByw(self) -> float:
        """
        Gets the value for byw.

        Returns:
            float: The current value.
        """
        ...
    def setPl(self, value: float) -> None:
        """
        Sets the value for pl.

        Args:
            value (float): The value to set.
        """
        ...
    def getPl(self) -> float:
        """
        Gets the value for pl.

        Returns:
            float: The current value.
        """
        ...

class Colorcorrect(FilterBase):
    """
    Adjust color white balance selectively for blacks and whites.
    """
    def setRl(self, value: float) -> None:
        """
        Sets the value for rl.

        Args:
            value (float): The value to set.
        """
        ...
    def getRl(self) -> float:
        """
        Gets the value for rl.

        Returns:
            float: The current value.
        """
        ...
    def setBl(self, value: float) -> None:
        """
        Sets the value for bl.

        Args:
            value (float): The value to set.
        """
        ...
    def getBl(self) -> float:
        """
        Gets the value for bl.

        Returns:
            float: The current value.
        """
        ...
    def setRh(self, value: float) -> None:
        """
        Sets the value for rh.

        Args:
            value (float): The value to set.
        """
        ...
    def getRh(self) -> float:
        """
        Gets the value for rh.

        Returns:
            float: The current value.
        """
        ...
    def setBh(self, value: float) -> None:
        """
        Sets the value for bh.

        Args:
            value (float): The value to set.
        """
        ...
    def getBh(self) -> float:
        """
        Gets the value for bh.

        Returns:
            float: The current value.
        """
        ...
    def setSaturation(self, value: float) -> None:
        """
        Sets the value for saturation.

        Args:
            value (float): The value to set.
        """
        ...
    def getSaturation(self) -> float:
        """
        Gets the value for saturation.

        Returns:
            float: The current value.
        """
        ...
    def setAnalyze(self, value: int) -> None:
        """
        Sets the value for analyze.

        Args:
            value (int): The value to set.
        """
        ...
    def getAnalyze(self) -> int:
        """
        Gets the value for analyze.

        Returns:
            int: The current value.
        """
        ...

class Colorize(FilterBase):
    """
    Overlay a solid color on the video stream.
    """
    def setHue(self, value: float) -> None:
        """
        Sets the value for hue.

        Args:
            value (float): The value to set.
        """
        ...
    def getHue(self) -> float:
        """
        Gets the value for hue.

        Returns:
            float: The current value.
        """
        ...
    def setSaturation(self, value: float) -> None:
        """
        Sets the value for saturation.

        Args:
            value (float): The value to set.
        """
        ...
    def getSaturation(self) -> float:
        """
        Gets the value for saturation.

        Returns:
            float: The current value.
        """
        ...
    def setLightness(self, value: float) -> None:
        """
        Sets the value for lightness.

        Args:
            value (float): The value to set.
        """
        ...
    def getLightness(self) -> float:
        """
        Gets the value for lightness.

        Returns:
            float: The current value.
        """
        ...
    def setMix(self, value: float) -> None:
        """
        Sets the value for mix.

        Args:
            value (float): The value to set.
        """
        ...
    def getMix(self) -> float:
        """
        Gets the value for mix.

        Returns:
            float: The current value.
        """
        ...

class Colorkey(FilterBase):
    """
    Turns a certain color into transparency. Operates on RGB colors.
    """
    def setColor(self, value: Any) -> None:
        """
        Sets the value for color.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColor(self) -> Any:
        """
        Gets the value for color.

        Returns:
            Any: The current value.
        """
        ...
    def setSimilarity(self, value: float) -> None:
        """
        Sets the value for similarity.

        Args:
            value (float): The value to set.
        """
        ...
    def getSimilarity(self) -> float:
        """
        Gets the value for similarity.

        Returns:
            float: The current value.
        """
        ...
    def setBlend(self, value: float) -> None:
        """
        Sets the value for blend.

        Args:
            value (float): The value to set.
        """
        ...
    def getBlend(self) -> float:
        """
        Gets the value for blend.

        Returns:
            float: The current value.
        """
        ...

class Colorhold(FilterBase):
    """
    Turns a certain color range into gray. Operates on RGB colors.
    """
    def setColor(self, value: Any) -> None:
        """
        Sets the value for color.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColor(self) -> Any:
        """
        Gets the value for color.

        Returns:
            Any: The current value.
        """
        ...
    def setSimilarity(self, value: float) -> None:
        """
        Sets the value for similarity.

        Args:
            value (float): The value to set.
        """
        ...
    def getSimilarity(self) -> float:
        """
        Gets the value for similarity.

        Returns:
            float: The current value.
        """
        ...
    def setBlend(self, value: float) -> None:
        """
        Sets the value for blend.

        Args:
            value (float): The value to set.
        """
        ...
    def getBlend(self) -> float:
        """
        Gets the value for blend.

        Returns:
            float: The current value.
        """
        ...

class Colorlevels(FilterBase):
    """
    Adjust the color levels.
    """
    def setRimin(self, value: float) -> None:
        """
        Sets the value for rimin.

        Args:
            value (float): The value to set.
        """
        ...
    def getRimin(self) -> float:
        """
        Gets the value for rimin.

        Returns:
            float: The current value.
        """
        ...
    def setGimin(self, value: float) -> None:
        """
        Sets the value for gimin.

        Args:
            value (float): The value to set.
        """
        ...
    def getGimin(self) -> float:
        """
        Gets the value for gimin.

        Returns:
            float: The current value.
        """
        ...
    def setBimin(self, value: float) -> None:
        """
        Sets the value for bimin.

        Args:
            value (float): The value to set.
        """
        ...
    def getBimin(self) -> float:
        """
        Gets the value for bimin.

        Returns:
            float: The current value.
        """
        ...
    def setAimin(self, value: float) -> None:
        """
        Sets the value for aimin.

        Args:
            value (float): The value to set.
        """
        ...
    def getAimin(self) -> float:
        """
        Gets the value for aimin.

        Returns:
            float: The current value.
        """
        ...
    def setRimax(self, value: float) -> None:
        """
        Sets the value for rimax.

        Args:
            value (float): The value to set.
        """
        ...
    def getRimax(self) -> float:
        """
        Gets the value for rimax.

        Returns:
            float: The current value.
        """
        ...
    def setGimax(self, value: float) -> None:
        """
        Sets the value for gimax.

        Args:
            value (float): The value to set.
        """
        ...
    def getGimax(self) -> float:
        """
        Gets the value for gimax.

        Returns:
            float: The current value.
        """
        ...
    def setBimax(self, value: float) -> None:
        """
        Sets the value for bimax.

        Args:
            value (float): The value to set.
        """
        ...
    def getBimax(self) -> float:
        """
        Gets the value for bimax.

        Returns:
            float: The current value.
        """
        ...
    def setAimax(self, value: float) -> None:
        """
        Sets the value for aimax.

        Args:
            value (float): The value to set.
        """
        ...
    def getAimax(self) -> float:
        """
        Gets the value for aimax.

        Returns:
            float: The current value.
        """
        ...
    def setRomin(self, value: float) -> None:
        """
        Sets the value for romin.

        Args:
            value (float): The value to set.
        """
        ...
    def getRomin(self) -> float:
        """
        Gets the value for romin.

        Returns:
            float: The current value.
        """
        ...
    def setGomin(self, value: float) -> None:
        """
        Sets the value for gomin.

        Args:
            value (float): The value to set.
        """
        ...
    def getGomin(self) -> float:
        """
        Gets the value for gomin.

        Returns:
            float: The current value.
        """
        ...
    def setBomin(self, value: float) -> None:
        """
        Sets the value for bomin.

        Args:
            value (float): The value to set.
        """
        ...
    def getBomin(self) -> float:
        """
        Gets the value for bomin.

        Returns:
            float: The current value.
        """
        ...
    def setAomin(self, value: float) -> None:
        """
        Sets the value for aomin.

        Args:
            value (float): The value to set.
        """
        ...
    def getAomin(self) -> float:
        """
        Gets the value for aomin.

        Returns:
            float: The current value.
        """
        ...
    def setRomax(self, value: float) -> None:
        """
        Sets the value for romax.

        Args:
            value (float): The value to set.
        """
        ...
    def getRomax(self) -> float:
        """
        Gets the value for romax.

        Returns:
            float: The current value.
        """
        ...
    def setGomax(self, value: float) -> None:
        """
        Sets the value for gomax.

        Args:
            value (float): The value to set.
        """
        ...
    def getGomax(self) -> float:
        """
        Gets the value for gomax.

        Returns:
            float: The current value.
        """
        ...
    def setBomax(self, value: float) -> None:
        """
        Sets the value for bomax.

        Args:
            value (float): The value to set.
        """
        ...
    def getBomax(self) -> float:
        """
        Gets the value for bomax.

        Returns:
            float: The current value.
        """
        ...
    def setAomax(self, value: float) -> None:
        """
        Sets the value for aomax.

        Args:
            value (float): The value to set.
        """
        ...
    def getAomax(self) -> float:
        """
        Gets the value for aomax.

        Returns:
            float: The current value.
        """
        ...
    def setPreserve(self, value: int) -> None:
        """
        Sets the value for preserve.

        Args:
            value (int): The value to set.
        """
        ...
    def getPreserve(self) -> int:
        """
        Gets the value for preserve.

        Returns:
            int: The current value.
        """
        ...

class Colormap(FilterBase):
    """
    Apply custom Color Maps to video stream.
    """
    def setPatch_size(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for patch_size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getPatch_size(self) -> Tuple[int, int]:
        """
        Gets the value for patch_size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setNb_patches(self, value: int) -> None:
        """
        Sets the value for nb_patches.

        Args:
            value (int): The value to set.
        """
        ...
    def getNb_patches(self) -> int:
        """
        Gets the value for nb_patches.

        Returns:
            int: The current value.
        """
        ...
    def setType(self, value: int) -> None:
        """
        Sets the value for type.

        Args:
            value (int): The value to set.
        """
        ...
    def getType(self) -> int:
        """
        Gets the value for type.

        Returns:
            int: The current value.
        """
        ...
    def setKernel(self, value: int) -> None:
        """
        Sets the value for kernel.

        Args:
            value (int): The value to set.
        """
        ...
    def getKernel(self) -> int:
        """
        Gets the value for kernel.

        Returns:
            int: The current value.
        """
        ...

class Colorspace(FilterBase):
    """
    Convert between colorspaces.
    """
    def setAll(self, value: int) -> None:
        """
        Sets the value for all.

        Args:
            value (int): The value to set.
        """
        ...
    def getAll(self) -> int:
        """
        Gets the value for all.

        Returns:
            int: The current value.
        """
        ...
    def setSpace(self, value: int) -> None:
        """
        Sets the value for space.

        Args:
            value (int): The value to set.
        """
        ...
    def getSpace(self) -> int:
        """
        Gets the value for space.

        Returns:
            int: The current value.
        """
        ...
    def setRange(self, value: int) -> None:
        """
        Sets the value for range.

        Args:
            value (int): The value to set.
        """
        ...
    def getRange(self) -> int:
        """
        Gets the value for range.

        Returns:
            int: The current value.
        """
        ...
    def setPrimaries(self, value: int) -> None:
        """
        Sets the value for primaries.

        Args:
            value (int): The value to set.
        """
        ...
    def getPrimaries(self) -> int:
        """
        Gets the value for primaries.

        Returns:
            int: The current value.
        """
        ...
    def setTrc(self, value: int) -> None:
        """
        Sets the value for trc.

        Args:
            value (int): The value to set.
        """
        ...
    def getTrc(self) -> int:
        """
        Gets the value for trc.

        Returns:
            int: The current value.
        """
        ...
    def setFormat(self, value: int) -> None:
        """
        Sets the value for format.

        Args:
            value (int): The value to set.
        """
        ...
    def getFormat(self) -> int:
        """
        Gets the value for format.

        Returns:
            int: The current value.
        """
        ...
    def setFast(self, value: bool) -> None:
        """
        Sets the value for fast.

        Args:
            value (bool): The value to set.
        """
        ...
    def getFast(self) -> bool:
        """
        Gets the value for fast.

        Returns:
            bool: The current value.
        """
        ...
    def setDither(self, value: int) -> None:
        """
        Sets the value for dither.

        Args:
            value (int): The value to set.
        """
        ...
    def getDither(self) -> int:
        """
        Gets the value for dither.

        Returns:
            int: The current value.
        """
        ...
    def setWpadapt(self, value: int) -> None:
        """
        Sets the value for wpadapt.

        Args:
            value (int): The value to set.
        """
        ...
    def getWpadapt(self) -> int:
        """
        Gets the value for wpadapt.

        Returns:
            int: The current value.
        """
        ...
    def setIall(self, value: int) -> None:
        """
        Sets the value for iall.

        Args:
            value (int): The value to set.
        """
        ...
    def getIall(self) -> int:
        """
        Gets the value for iall.

        Returns:
            int: The current value.
        """
        ...
    def setIspace(self, value: int) -> None:
        """
        Sets the value for ispace.

        Args:
            value (int): The value to set.
        """
        ...
    def getIspace(self) -> int:
        """
        Gets the value for ispace.

        Returns:
            int: The current value.
        """
        ...
    def setIrange(self, value: int) -> None:
        """
        Sets the value for irange.

        Args:
            value (int): The value to set.
        """
        ...
    def getIrange(self) -> int:
        """
        Gets the value for irange.

        Returns:
            int: The current value.
        """
        ...
    def setIprimaries(self, value: int) -> None:
        """
        Sets the value for iprimaries.

        Args:
            value (int): The value to set.
        """
        ...
    def getIprimaries(self) -> int:
        """
        Gets the value for iprimaries.

        Returns:
            int: The current value.
        """
        ...
    def setItrc(self, value: int) -> None:
        """
        Sets the value for itrc.

        Args:
            value (int): The value to set.
        """
        ...
    def getItrc(self) -> int:
        """
        Gets the value for itrc.

        Returns:
            int: The current value.
        """
        ...

class Colortemperature(FilterBase):
    """
    Adjust color temperature of video.
    """
    def setTemperature(self, value: float) -> None:
        """
        Sets the value for temperature.

        Args:
            value (float): The value to set.
        """
        ...
    def getTemperature(self) -> float:
        """
        Gets the value for temperature.

        Returns:
            float: The current value.
        """
        ...
    def setMix(self, value: float) -> None:
        """
        Sets the value for mix.

        Args:
            value (float): The value to set.
        """
        ...
    def getMix(self) -> float:
        """
        Gets the value for mix.

        Returns:
            float: The current value.
        """
        ...
    def setPl(self, value: float) -> None:
        """
        Sets the value for pl.

        Args:
            value (float): The value to set.
        """
        ...
    def getPl(self) -> float:
        """
        Gets the value for pl.

        Returns:
            float: The current value.
        """
        ...

class Convolution(FilterBase):
    """
    Apply convolution filter.
    """
    def set_0m(self, value: Any) -> None:
        """
        Sets the value for _0m.

        Args:
            value (Any): The value to set.
        """
        ...
    def get_0m(self) -> Any:
        """
        Gets the value for _0m.

        Returns:
            Any: The current value.
        """
        ...
    def set_1m(self, value: Any) -> None:
        """
        Sets the value for _1m.

        Args:
            value (Any): The value to set.
        """
        ...
    def get_1m(self) -> Any:
        """
        Gets the value for _1m.

        Returns:
            Any: The current value.
        """
        ...
    def set_2m(self, value: Any) -> None:
        """
        Sets the value for _2m.

        Args:
            value (Any): The value to set.
        """
        ...
    def get_2m(self) -> Any:
        """
        Gets the value for _2m.

        Returns:
            Any: The current value.
        """
        ...
    def set_3m(self, value: Any) -> None:
        """
        Sets the value for _3m.

        Args:
            value (Any): The value to set.
        """
        ...
    def get_3m(self) -> Any:
        """
        Gets the value for _3m.

        Returns:
            Any: The current value.
        """
        ...
    def set_0rdiv(self, value: float) -> None:
        """
        Sets the value for _0rdiv.

        Args:
            value (float): The value to set.
        """
        ...
    def get_0rdiv(self) -> float:
        """
        Gets the value for _0rdiv.

        Returns:
            float: The current value.
        """
        ...
    def set_1rdiv(self, value: float) -> None:
        """
        Sets the value for _1rdiv.

        Args:
            value (float): The value to set.
        """
        ...
    def get_1rdiv(self) -> float:
        """
        Gets the value for _1rdiv.

        Returns:
            float: The current value.
        """
        ...
    def set_2rdiv(self, value: float) -> None:
        """
        Sets the value for _2rdiv.

        Args:
            value (float): The value to set.
        """
        ...
    def get_2rdiv(self) -> float:
        """
        Gets the value for _2rdiv.

        Returns:
            float: The current value.
        """
        ...
    def set_3rdiv(self, value: float) -> None:
        """
        Sets the value for _3rdiv.

        Args:
            value (float): The value to set.
        """
        ...
    def get_3rdiv(self) -> float:
        """
        Gets the value for _3rdiv.

        Returns:
            float: The current value.
        """
        ...
    def set_0bias(self, value: float) -> None:
        """
        Sets the value for _0bias.

        Args:
            value (float): The value to set.
        """
        ...
    def get_0bias(self) -> float:
        """
        Gets the value for _0bias.

        Returns:
            float: The current value.
        """
        ...
    def set_1bias(self, value: float) -> None:
        """
        Sets the value for _1bias.

        Args:
            value (float): The value to set.
        """
        ...
    def get_1bias(self) -> float:
        """
        Gets the value for _1bias.

        Returns:
            float: The current value.
        """
        ...
    def set_2bias(self, value: float) -> None:
        """
        Sets the value for _2bias.

        Args:
            value (float): The value to set.
        """
        ...
    def get_2bias(self) -> float:
        """
        Gets the value for _2bias.

        Returns:
            float: The current value.
        """
        ...
    def set_3bias(self, value: float) -> None:
        """
        Sets the value for _3bias.

        Args:
            value (float): The value to set.
        """
        ...
    def get_3bias(self) -> float:
        """
        Gets the value for _3bias.

        Returns:
            float: The current value.
        """
        ...
    def set_0mode(self, value: int) -> None:
        """
        Sets the value for _0mode.

        Args:
            value (int): The value to set.
        """
        ...
    def get_0mode(self) -> int:
        """
        Gets the value for _0mode.

        Returns:
            int: The current value.
        """
        ...
    def set_1mode(self, value: int) -> None:
        """
        Sets the value for _1mode.

        Args:
            value (int): The value to set.
        """
        ...
    def get_1mode(self) -> int:
        """
        Gets the value for _1mode.

        Returns:
            int: The current value.
        """
        ...
    def set_2mode(self, value: int) -> None:
        """
        Sets the value for _2mode.

        Args:
            value (int): The value to set.
        """
        ...
    def get_2mode(self) -> int:
        """
        Gets the value for _2mode.

        Returns:
            int: The current value.
        """
        ...
    def set_3mode(self, value: int) -> None:
        """
        Sets the value for _3mode.

        Args:
            value (int): The value to set.
        """
        ...
    def get_3mode(self) -> int:
        """
        Gets the value for _3mode.

        Returns:
            int: The current value.
        """
        ...

class Convolve(FilterBase):
    """
    Convolve first video stream with second video stream.
    """
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setImpulse(self, value: int) -> None:
        """
        Sets the value for impulse.

        Args:
            value (int): The value to set.
        """
        ...
    def getImpulse(self) -> int:
        """
        Gets the value for impulse.

        Returns:
            int: The current value.
        """
        ...
    def setNoise(self, value: float) -> None:
        """
        Sets the value for noise.

        Args:
            value (float): The value to set.
        """
        ...
    def getNoise(self) -> float:
        """
        Gets the value for noise.

        Returns:
            float: The current value.
        """
        ...

class Copy(FilterBase):
    """
    Copy the input video unchanged to the output.
    """
    pass

class Corr(FilterBase):
    """
    Calculate the correlation between two video streams.
    """
    pass

class Crop(FilterBase):
    """
    Crop the input video.
    """
    def setOut_w(self, value: Any) -> None:
        """
        Sets the value for out_w.

        Args:
            value (Any): The value to set.
        """
        ...
    def getOut_w(self) -> Any:
        """
        Gets the value for out_w.

        Returns:
            Any: The current value.
        """
        ...
    def setOut_h(self, value: Any) -> None:
        """
        Sets the value for out_h.

        Args:
            value (Any): The value to set.
        """
        ...
    def getOut_h(self) -> Any:
        """
        Gets the value for out_h.

        Returns:
            Any: The current value.
        """
        ...
    def setXCropArea(self, value: Any) -> None:
        """
        Sets the value for xcroparea.

        Args:
            value (Any): The value to set.
        """
        ...
    def getXCropArea(self) -> Any:
        """
        Gets the value for xcroparea.

        Returns:
            Any: The current value.
        """
        ...
    def setYCropArea(self, value: Any) -> None:
        """
        Sets the value for ycroparea.

        Args:
            value (Any): The value to set.
        """
        ...
    def getYCropArea(self) -> Any:
        """
        Gets the value for ycroparea.

        Returns:
            Any: The current value.
        """
        ...
    def setKeep_aspect(self, value: bool) -> None:
        """
        Sets the value for keep_aspect.

        Args:
            value (bool): The value to set.
        """
        ...
    def getKeep_aspect(self) -> bool:
        """
        Gets the value for keep_aspect.

        Returns:
            bool: The current value.
        """
        ...
    def setExact(self, value: bool) -> None:
        """
        Sets the value for exact.

        Args:
            value (bool): The value to set.
        """
        ...
    def getExact(self) -> bool:
        """
        Gets the value for exact.

        Returns:
            bool: The current value.
        """
        ...

class Curves(FilterBase):
    """
    Adjust components curves.
    """
    def setPreset(self, value: int) -> None:
        """
        Sets the value for preset.

        Args:
            value (int): The value to set.
        """
        ...
    def getPreset(self) -> int:
        """
        Gets the value for preset.

        Returns:
            int: The current value.
        """
        ...
    def setMaster(self, value: Any) -> None:
        """
        Sets the value for master.

        Args:
            value (Any): The value to set.
        """
        ...
    def getMaster(self) -> Any:
        """
        Gets the value for master.

        Returns:
            Any: The current value.
        """
        ...
    def setRed(self, value: Any) -> None:
        """
        Sets the value for red.

        Args:
            value (Any): The value to set.
        """
        ...
    def getRed(self) -> Any:
        """
        Gets the value for red.

        Returns:
            Any: The current value.
        """
        ...
    def setGreen(self, value: Any) -> None:
        """
        Sets the value for green.

        Args:
            value (Any): The value to set.
        """
        ...
    def getGreen(self) -> Any:
        """
        Gets the value for green.

        Returns:
            Any: The current value.
        """
        ...
    def setBlue(self, value: Any) -> None:
        """
        Sets the value for blue.

        Args:
            value (Any): The value to set.
        """
        ...
    def getBlue(self) -> Any:
        """
        Gets the value for blue.

        Returns:
            Any: The current value.
        """
        ...
    def setAll(self, value: Any) -> None:
        """
        Sets the value for all.

        Args:
            value (Any): The value to set.
        """
        ...
    def getAll(self) -> Any:
        """
        Gets the value for all.

        Returns:
            Any: The current value.
        """
        ...
    def setPsfile(self, value: Any) -> None:
        """
        Sets the value for psfile.

        Args:
            value (Any): The value to set.
        """
        ...
    def getPsfile(self) -> Any:
        """
        Gets the value for psfile.

        Returns:
            Any: The current value.
        """
        ...
    def setPlot(self, value: Any) -> None:
        """
        Sets the value for plot.

        Args:
            value (Any): The value to set.
        """
        ...
    def getPlot(self) -> Any:
        """
        Gets the value for plot.

        Returns:
            Any: The current value.
        """
        ...
    def setInterp(self, value: int) -> None:
        """
        Sets the value for interp.

        Args:
            value (int): The value to set.
        """
        ...
    def getInterp(self) -> int:
        """
        Gets the value for interp.

        Returns:
            int: The current value.
        """
        ...

class Datascope(FilterBase):
    """
    Video data analysis.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setXOffset(self, value: int) -> None:
        """
        Sets the value for xoffset.

        Args:
            value (int): The value to set.
        """
        ...
    def getXOffset(self) -> int:
        """
        Gets the value for xoffset.

        Returns:
            int: The current value.
        """
        ...
    def setYOffset(self, value: int) -> None:
        """
        Sets the value for yoffset.

        Args:
            value (int): The value to set.
        """
        ...
    def getYOffset(self) -> int:
        """
        Gets the value for yoffset.

        Returns:
            int: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setAxis(self, value: bool) -> None:
        """
        Sets the value for axis.

        Args:
            value (bool): The value to set.
        """
        ...
    def getAxis(self) -> bool:
        """
        Gets the value for axis.

        Returns:
            bool: The current value.
        """
        ...
    def setOpacity(self, value: float) -> None:
        """
        Sets the value for opacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getOpacity(self) -> float:
        """
        Gets the value for opacity.

        Returns:
            float: The current value.
        """
        ...
    def setFormat(self, value: int) -> None:
        """
        Sets the value for format.

        Args:
            value (int): The value to set.
        """
        ...
    def getFormat(self) -> int:
        """
        Gets the value for format.

        Returns:
            int: The current value.
        """
        ...
    def setComponents(self, value: int) -> None:
        """
        Sets the value for components.

        Args:
            value (int): The value to set.
        """
        ...
    def getComponents(self) -> int:
        """
        Gets the value for components.

        Returns:
            int: The current value.
        """
        ...

class Dblur(FilterBase):
    """
    Apply Directional Blur filter.
    """
    def setAngle(self, value: float) -> None:
        """
        Sets the value for angle.

        Args:
            value (float): The value to set.
        """
        ...
    def getAngle(self) -> float:
        """
        Gets the value for angle.

        Returns:
            float: The current value.
        """
        ...
    def setRadius(self, value: float) -> None:
        """
        Sets the value for radius.

        Args:
            value (float): The value to set.
        """
        ...
    def getRadius(self) -> float:
        """
        Gets the value for radius.

        Returns:
            float: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Dctdnoiz(FilterBase):
    """
    Denoise frames using 2D DCT.
    """
    def setSigma(self, value: float) -> None:
        """
        Sets the value for sigma.

        Args:
            value (float): The value to set.
        """
        ...
    def getSigma(self) -> float:
        """
        Gets the value for sigma.

        Returns:
            float: The current value.
        """
        ...
    def setOverlap(self, value: int) -> None:
        """
        Sets the value for overlap.

        Args:
            value (int): The value to set.
        """
        ...
    def getOverlap(self) -> int:
        """
        Gets the value for overlap.

        Returns:
            int: The current value.
        """
        ...
    def setExpr(self, value: Any) -> None:
        """
        Sets the value for expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getExpr(self) -> Any:
        """
        Gets the value for expr.

        Returns:
            Any: The current value.
        """
        ...
    def setBlockSizeExpressedInBits(self, value: int) -> None:
        """
        Sets the value for blocksizeexpressedinbits.

        Args:
            value (int): The value to set.
        """
        ...
    def getBlockSizeExpressedInBits(self) -> int:
        """
        Gets the value for blocksizeexpressedinbits.

        Returns:
            int: The current value.
        """
        ...

class Deband(FilterBase):
    """
    Debands video.
    """
    def set_1thr(self, value: float) -> None:
        """
        Sets the value for _1thr.

        Args:
            value (float): The value to set.
        """
        ...
    def get_1thr(self) -> float:
        """
        Gets the value for _1thr.

        Returns:
            float: The current value.
        """
        ...
    def set_2thr(self, value: float) -> None:
        """
        Sets the value for _2thr.

        Args:
            value (float): The value to set.
        """
        ...
    def get_2thr(self) -> float:
        """
        Gets the value for _2thr.

        Returns:
            float: The current value.
        """
        ...
    def set_3thr(self, value: float) -> None:
        """
        Sets the value for _3thr.

        Args:
            value (float): The value to set.
        """
        ...
    def get_3thr(self) -> float:
        """
        Gets the value for _3thr.

        Returns:
            float: The current value.
        """
        ...
    def set_4thr(self, value: float) -> None:
        """
        Sets the value for _4thr.

        Args:
            value (float): The value to set.
        """
        ...
    def get_4thr(self) -> float:
        """
        Gets the value for _4thr.

        Returns:
            float: The current value.
        """
        ...
    def setRange(self, value: int) -> None:
        """
        Sets the value for range.

        Args:
            value (int): The value to set.
        """
        ...
    def getRange(self) -> int:
        """
        Gets the value for range.

        Returns:
            int: The current value.
        """
        ...
    def setDirection(self, value: float) -> None:
        """
        Sets the value for direction.

        Args:
            value (float): The value to set.
        """
        ...
    def getDirection(self) -> float:
        """
        Gets the value for direction.

        Returns:
            float: The current value.
        """
        ...
    def setBlur(self, value: bool) -> None:
        """
        Sets the value for blur.

        Args:
            value (bool): The value to set.
        """
        ...
    def getBlur(self) -> bool:
        """
        Gets the value for blur.

        Returns:
            bool: The current value.
        """
        ...
    def setCoupling(self, value: bool) -> None:
        """
        Sets the value for coupling.

        Args:
            value (bool): The value to set.
        """
        ...
    def getCoupling(self) -> bool:
        """
        Gets the value for coupling.

        Returns:
            bool: The current value.
        """
        ...

class Deblock(FilterBase):
    """
    Deblock video.
    """
    def setFilter(self, value: int) -> None:
        """
        Sets the value for filter.

        Args:
            value (int): The value to set.
        """
        ...
    def getFilter(self) -> int:
        """
        Gets the value for filter.

        Returns:
            int: The current value.
        """
        ...
    def setBlock(self, value: int) -> None:
        """
        Sets the value for block.

        Args:
            value (int): The value to set.
        """
        ...
    def getBlock(self) -> int:
        """
        Gets the value for block.

        Returns:
            int: The current value.
        """
        ...
    def setAlpha(self, value: float) -> None:
        """
        Sets the value for alpha.

        Args:
            value (float): The value to set.
        """
        ...
    def getAlpha(self) -> float:
        """
        Gets the value for alpha.

        Returns:
            float: The current value.
        """
        ...
    def setBeta(self, value: float) -> None:
        """
        Sets the value for beta.

        Args:
            value (float): The value to set.
        """
        ...
    def getBeta(self) -> float:
        """
        Gets the value for beta.

        Returns:
            float: The current value.
        """
        ...
    def setGamma(self, value: float) -> None:
        """
        Sets the value for gamma.

        Args:
            value (float): The value to set.
        """
        ...
    def getGamma(self) -> float:
        """
        Gets the value for gamma.

        Returns:
            float: The current value.
        """
        ...
    def setDelta(self, value: float) -> None:
        """
        Sets the value for delta.

        Args:
            value (float): The value to set.
        """
        ...
    def getDelta(self) -> float:
        """
        Gets the value for delta.

        Returns:
            float: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Decimate(FilterBase):
    """
    Decimate frames (post field matching filter).
    """
    def setCycle(self, value: int) -> None:
        """
        Sets the value for cycle.

        Args:
            value (int): The value to set.
        """
        ...
    def getCycle(self) -> int:
        """
        Gets the value for cycle.

        Returns:
            int: The current value.
        """
        ...
    def setDupthresh(self, value: float) -> None:
        """
        Sets the value for dupthresh.

        Args:
            value (float): The value to set.
        """
        ...
    def getDupthresh(self) -> float:
        """
        Gets the value for dupthresh.

        Returns:
            float: The current value.
        """
        ...
    def setScthresh(self, value: float) -> None:
        """
        Sets the value for scthresh.

        Args:
            value (float): The value to set.
        """
        ...
    def getScthresh(self) -> float:
        """
        Gets the value for scthresh.

        Returns:
            float: The current value.
        """
        ...
    def setBlockx(self, value: int) -> None:
        """
        Sets the value for blockx.

        Args:
            value (int): The value to set.
        """
        ...
    def getBlockx(self) -> int:
        """
        Gets the value for blockx.

        Returns:
            int: The current value.
        """
        ...
    def setBlocky(self, value: int) -> None:
        """
        Sets the value for blocky.

        Args:
            value (int): The value to set.
        """
        ...
    def getBlocky(self) -> int:
        """
        Gets the value for blocky.

        Returns:
            int: The current value.
        """
        ...
    def setPpsrc(self, value: bool) -> None:
        """
        Sets the value for ppsrc.

        Args:
            value (bool): The value to set.
        """
        ...
    def getPpsrc(self) -> bool:
        """
        Gets the value for ppsrc.

        Returns:
            bool: The current value.
        """
        ...
    def setChroma(self, value: bool) -> None:
        """
        Sets the value for chroma.

        Args:
            value (bool): The value to set.
        """
        ...
    def getChroma(self) -> bool:
        """
        Gets the value for chroma.

        Returns:
            bool: The current value.
        """
        ...
    def setMixed(self, value: bool) -> None:
        """
        Sets the value for mixed.

        Args:
            value (bool): The value to set.
        """
        ...
    def getMixed(self) -> bool:
        """
        Gets the value for mixed.

        Returns:
            bool: The current value.
        """
        ...

class Deconvolve(FilterBase):
    """
    Deconvolve first video stream with second video stream.
    """
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setImpulse(self, value: int) -> None:
        """
        Sets the value for impulse.

        Args:
            value (int): The value to set.
        """
        ...
    def getImpulse(self) -> int:
        """
        Gets the value for impulse.

        Returns:
            int: The current value.
        """
        ...
    def setNoise(self, value: float) -> None:
        """
        Sets the value for noise.

        Args:
            value (float): The value to set.
        """
        ...
    def getNoise(self) -> float:
        """
        Gets the value for noise.

        Returns:
            float: The current value.
        """
        ...

class Dedot(FilterBase):
    """
    Reduce cross-luminance and cross-color.
    """
    def setFilteringMode(self, value: int) -> None:
        """
        Sets the value for filteringmode.

        Args:
            value (int): The value to set.
        """
        ...
    def getFilteringMode(self) -> int:
        """
        Gets the value for filteringmode.

        Returns:
            int: The current value.
        """
        ...
    def setLt(self, value: float) -> None:
        """
        Sets the value for lt.

        Args:
            value (float): The value to set.
        """
        ...
    def getLt(self) -> float:
        """
        Gets the value for lt.

        Returns:
            float: The current value.
        """
        ...
    def setTl(self, value: float) -> None:
        """
        Sets the value for tl.

        Args:
            value (float): The value to set.
        """
        ...
    def getTl(self) -> float:
        """
        Gets the value for tl.

        Returns:
            float: The current value.
        """
        ...
    def setTc(self, value: float) -> None:
        """
        Sets the value for tc.

        Args:
            value (float): The value to set.
        """
        ...
    def getTc(self) -> float:
        """
        Gets the value for tc.

        Returns:
            float: The current value.
        """
        ...
    def setCt(self, value: float) -> None:
        """
        Sets the value for ct.

        Args:
            value (float): The value to set.
        """
        ...
    def getCt(self) -> float:
        """
        Gets the value for ct.

        Returns:
            float: The current value.
        """
        ...

class Deflate(FilterBase):
    """
    Apply deflate effect.
    """
    def setThreshold0(self, value: int) -> None:
        """
        Sets the value for threshold0.

        Args:
            value (int): The value to set.
        """
        ...
    def getThreshold0(self) -> int:
        """
        Gets the value for threshold0.

        Returns:
            int: The current value.
        """
        ...
    def setThreshold1(self, value: int) -> None:
        """
        Sets the value for threshold1.

        Args:
            value (int): The value to set.
        """
        ...
    def getThreshold1(self) -> int:
        """
        Gets the value for threshold1.

        Returns:
            int: The current value.
        """
        ...
    def setThreshold2(self, value: int) -> None:
        """
        Sets the value for threshold2.

        Args:
            value (int): The value to set.
        """
        ...
    def getThreshold2(self) -> int:
        """
        Gets the value for threshold2.

        Returns:
            int: The current value.
        """
        ...
    def setThreshold3(self, value: int) -> None:
        """
        Sets the value for threshold3.

        Args:
            value (int): The value to set.
        """
        ...
    def getThreshold3(self) -> int:
        """
        Gets the value for threshold3.

        Returns:
            int: The current value.
        """
        ...

class Deflicker(FilterBase):
    """
    Remove temporal frame luminance variations.
    """
    def setSize(self, value: int) -> None:
        """
        Sets the value for size.

        Args:
            value (int): The value to set.
        """
        ...
    def getSize(self) -> int:
        """
        Gets the value for size.

        Returns:
            int: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setBypass(self, value: bool) -> None:
        """
        Sets the value for bypass.

        Args:
            value (bool): The value to set.
        """
        ...
    def getBypass(self) -> bool:
        """
        Gets the value for bypass.

        Returns:
            bool: The current value.
        """
        ...

class Dejudder(FilterBase):
    """
    Remove judder produced by pullup.
    """
    def setCycle(self, value: int) -> None:
        """
        Sets the value for cycle.

        Args:
            value (int): The value to set.
        """
        ...
    def getCycle(self) -> int:
        """
        Gets the value for cycle.

        Returns:
            int: The current value.
        """
        ...

class Derain(FilterBase):
    """
    Apply derain filter to the input.
    """
    def setFilter_type(self, value: int) -> None:
        """
        Sets the value for filter_type.

        Args:
            value (int): The value to set.
        """
        ...
    def getFilter_type(self) -> int:
        """
        Gets the value for filter_type.

        Returns:
            int: The current value.
        """
        ...
    def setDnn_backend(self, value: int) -> None:
        """
        Sets the value for dnn_backend.

        Args:
            value (int): The value to set.
        """
        ...
    def getDnn_backend(self) -> int:
        """
        Gets the value for dnn_backend.

        Returns:
            int: The current value.
        """
        ...
    def setModel(self, value: Any) -> None:
        """
        Sets the value for model.

        Args:
            value (Any): The value to set.
        """
        ...
    def getModel(self) -> Any:
        """
        Gets the value for model.

        Returns:
            Any: The current value.
        """
        ...
    def setInput(self, value: Any) -> None:
        """
        Sets the value for input.

        Args:
            value (Any): The value to set.
        """
        ...
    def getInput(self) -> Any:
        """
        Gets the value for input.

        Returns:
            Any: The current value.
        """
        ...
    def setOutput(self, value: Any) -> None:
        """
        Sets the value for output.

        Args:
            value (Any): The value to set.
        """
        ...
    def getOutput(self) -> Any:
        """
        Gets the value for output.

        Returns:
            Any: The current value.
        """
        ...

class Deshake(FilterBase):
    """
    Stabilize shaky video.
    """
    def setXForTheRectangularSearchArea(self, value: int) -> None:
        """
        Sets the value for xfortherectangularsearcharea.

        Args:
            value (int): The value to set.
        """
        ...
    def getXForTheRectangularSearchArea(self) -> int:
        """
        Gets the value for xfortherectangularsearcharea.

        Returns:
            int: The current value.
        """
        ...
    def setYForTheRectangularSearchArea(self, value: int) -> None:
        """
        Sets the value for yfortherectangularsearcharea.

        Args:
            value (int): The value to set.
        """
        ...
    def getYForTheRectangularSearchArea(self) -> int:
        """
        Gets the value for yfortherectangularsearcharea.

        Returns:
            int: The current value.
        """
        ...
    def setWidthForTheRectangularSearchArea(self, value: int) -> None:
        """
        Sets the value for widthfortherectangularsearcharea.

        Args:
            value (int): The value to set.
        """
        ...
    def getWidthForTheRectangularSearchArea(self) -> int:
        """
        Gets the value for widthfortherectangularsearcharea.

        Returns:
            int: The current value.
        """
        ...
    def setHeightForTheRectangularSearchArea(self, value: int) -> None:
        """
        Sets the value for heightfortherectangularsearcharea.

        Args:
            value (int): The value to set.
        """
        ...
    def getHeightForTheRectangularSearchArea(self) -> int:
        """
        Gets the value for heightfortherectangularsearcharea.

        Returns:
            int: The current value.
        """
        ...
    def setRx(self, value: int) -> None:
        """
        Sets the value for rx.

        Args:
            value (int): The value to set.
        """
        ...
    def getRx(self) -> int:
        """
        Gets the value for rx.

        Returns:
            int: The current value.
        """
        ...
    def setRy(self, value: int) -> None:
        """
        Sets the value for ry.

        Args:
            value (int): The value to set.
        """
        ...
    def getRy(self) -> int:
        """
        Gets the value for ry.

        Returns:
            int: The current value.
        """
        ...
    def setEdge(self, value: int) -> None:
        """
        Sets the value for edge.

        Args:
            value (int): The value to set.
        """
        ...
    def getEdge(self) -> int:
        """
        Gets the value for edge.

        Returns:
            int: The current value.
        """
        ...
    def setBlocksize(self, value: int) -> None:
        """
        Sets the value for blocksize.

        Args:
            value (int): The value to set.
        """
        ...
    def getBlocksize(self) -> int:
        """
        Gets the value for blocksize.

        Returns:
            int: The current value.
        """
        ...
    def setContrast(self, value: int) -> None:
        """
        Sets the value for contrast.

        Args:
            value (int): The value to set.
        """
        ...
    def getContrast(self) -> int:
        """
        Gets the value for contrast.

        Returns:
            int: The current value.
        """
        ...
    def setSearch(self, value: int) -> None:
        """
        Sets the value for search.

        Args:
            value (int): The value to set.
        """
        ...
    def getSearch(self) -> int:
        """
        Gets the value for search.

        Returns:
            int: The current value.
        """
        ...
    def setFilename(self, value: Any) -> None:
        """
        Sets the value for filename.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFilename(self) -> Any:
        """
        Gets the value for filename.

        Returns:
            Any: The current value.
        """
        ...
    def setOpencl(self, value: bool) -> None:
        """
        Sets the value for opencl.

        Args:
            value (bool): The value to set.
        """
        ...
    def getOpencl(self) -> bool:
        """
        Gets the value for opencl.

        Returns:
            bool: The current value.
        """
        ...

class Despill(FilterBase):
    """
    Despill video.
    """
    def setType(self, value: int) -> None:
        """
        Sets the value for type.

        Args:
            value (int): The value to set.
        """
        ...
    def getType(self) -> int:
        """
        Gets the value for type.

        Returns:
            int: The current value.
        """
        ...
    def setMix(self, value: float) -> None:
        """
        Sets the value for mix.

        Args:
            value (float): The value to set.
        """
        ...
    def getMix(self) -> float:
        """
        Gets the value for mix.

        Returns:
            float: The current value.
        """
        ...
    def setExpand(self, value: float) -> None:
        """
        Sets the value for expand.

        Args:
            value (float): The value to set.
        """
        ...
    def getExpand(self) -> float:
        """
        Gets the value for expand.

        Returns:
            float: The current value.
        """
        ...
    def setRed(self, value: float) -> None:
        """
        Sets the value for red.

        Args:
            value (float): The value to set.
        """
        ...
    def getRed(self) -> float:
        """
        Gets the value for red.

        Returns:
            float: The current value.
        """
        ...
    def setGreen(self, value: float) -> None:
        """
        Sets the value for green.

        Args:
            value (float): The value to set.
        """
        ...
    def getGreen(self) -> float:
        """
        Gets the value for green.

        Returns:
            float: The current value.
        """
        ...
    def setBlue(self, value: float) -> None:
        """
        Sets the value for blue.

        Args:
            value (float): The value to set.
        """
        ...
    def getBlue(self) -> float:
        """
        Gets the value for blue.

        Returns:
            float: The current value.
        """
        ...
    def setBrightness(self, value: float) -> None:
        """
        Sets the value for brightness.

        Args:
            value (float): The value to set.
        """
        ...
    def getBrightness(self) -> float:
        """
        Gets the value for brightness.

        Returns:
            float: The current value.
        """
        ...
    def setAlpha(self, value: bool) -> None:
        """
        Sets the value for alpha.

        Args:
            value (bool): The value to set.
        """
        ...
    def getAlpha(self) -> bool:
        """
        Gets the value for alpha.

        Returns:
            bool: The current value.
        """
        ...

class Detelecine(FilterBase):
    """
    Apply an inverse telecine pattern.
    """
    def setFirst_field(self, value: int) -> None:
        """
        Sets the value for first_field.

        Args:
            value (int): The value to set.
        """
        ...
    def getFirst_field(self) -> int:
        """
        Gets the value for first_field.

        Returns:
            int: The current value.
        """
        ...
    def setPattern(self, value: Any) -> None:
        """
        Sets the value for pattern.

        Args:
            value (Any): The value to set.
        """
        ...
    def getPattern(self) -> Any:
        """
        Gets the value for pattern.

        Returns:
            Any: The current value.
        """
        ...
    def setStart_frame(self, value: int) -> None:
        """
        Sets the value for start_frame.

        Args:
            value (int): The value to set.
        """
        ...
    def getStart_frame(self) -> int:
        """
        Gets the value for start_frame.

        Returns:
            int: The current value.
        """
        ...

class Dilation(FilterBase):
    """
    Apply dilation effect.
    """
    def setCoordinates(self, value: int) -> None:
        """
        Sets the value for coordinates.

        Args:
            value (int): The value to set.
        """
        ...
    def getCoordinates(self) -> int:
        """
        Gets the value for coordinates.

        Returns:
            int: The current value.
        """
        ...
    def setThreshold0(self, value: int) -> None:
        """
        Sets the value for threshold0.

        Args:
            value (int): The value to set.
        """
        ...
    def getThreshold0(self) -> int:
        """
        Gets the value for threshold0.

        Returns:
            int: The current value.
        """
        ...
    def setThreshold1(self, value: int) -> None:
        """
        Sets the value for threshold1.

        Args:
            value (int): The value to set.
        """
        ...
    def getThreshold1(self) -> int:
        """
        Gets the value for threshold1.

        Returns:
            int: The current value.
        """
        ...
    def setThreshold2(self, value: int) -> None:
        """
        Sets the value for threshold2.

        Args:
            value (int): The value to set.
        """
        ...
    def getThreshold2(self) -> int:
        """
        Gets the value for threshold2.

        Returns:
            int: The current value.
        """
        ...
    def setThreshold3(self, value: int) -> None:
        """
        Sets the value for threshold3.

        Args:
            value (int): The value to set.
        """
        ...
    def getThreshold3(self) -> int:
        """
        Gets the value for threshold3.

        Returns:
            int: The current value.
        """
        ...

class Displace(FilterBase):
    """
    Displace pixels.
    """
    def setEdge(self, value: int) -> None:
        """
        Sets the value for edge.

        Args:
            value (int): The value to set.
        """
        ...
    def getEdge(self) -> int:
        """
        Gets the value for edge.

        Returns:
            int: The current value.
        """
        ...

class Dnn_classify(FilterBase):
    """
    Apply DNN classify filter to the input.
    """
    def setDnn_backend(self, value: int) -> None:
        """
        Sets the value for dnn_backend.

        Args:
            value (int): The value to set.
        """
        ...
    def getDnn_backend(self) -> int:
        """
        Gets the value for dnn_backend.

        Returns:
            int: The current value.
        """
        ...
    def setModel(self, value: Any) -> None:
        """
        Sets the value for model.

        Args:
            value (Any): The value to set.
        """
        ...
    def getModel(self) -> Any:
        """
        Gets the value for model.

        Returns:
            Any: The current value.
        """
        ...
    def setInput(self, value: Any) -> None:
        """
        Sets the value for input.

        Args:
            value (Any): The value to set.
        """
        ...
    def getInput(self) -> Any:
        """
        Gets the value for input.

        Returns:
            Any: The current value.
        """
        ...
    def setOutput(self, value: Any) -> None:
        """
        Sets the value for output.

        Args:
            value (Any): The value to set.
        """
        ...
    def getOutput(self) -> Any:
        """
        Gets the value for output.

        Returns:
            Any: The current value.
        """
        ...
    def setBackend_configs(self, value: Any) -> None:
        """
        Sets the value for backend_configs.

        Args:
            value (Any): The value to set.
        """
        ...
    def getBackend_configs(self) -> Any:
        """
        Gets the value for backend_configs.

        Returns:
            Any: The current value.
        """
        ...
    def setAsync(self, value: bool) -> None:
        """
        Sets the value for async.

        Args:
            value (bool): The value to set.
        """
        ...
    def getAsync(self) -> bool:
        """
        Gets the value for async.

        Returns:
            bool: The current value.
        """
        ...
    def setConfidence(self, value: float) -> None:
        """
        Sets the value for confidence.

        Args:
            value (float): The value to set.
        """
        ...
    def getConfidence(self) -> float:
        """
        Gets the value for confidence.

        Returns:
            float: The current value.
        """
        ...
    def setLabels(self, value: Any) -> None:
        """
        Sets the value for labels.

        Args:
            value (Any): The value to set.
        """
        ...
    def getLabels(self) -> Any:
        """
        Gets the value for labels.

        Returns:
            Any: The current value.
        """
        ...
    def setTarget(self, value: Any) -> None:
        """
        Sets the value for target.

        Args:
            value (Any): The value to set.
        """
        ...
    def getTarget(self) -> Any:
        """
        Gets the value for target.

        Returns:
            Any: The current value.
        """
        ...

class Dnn_detect(FilterBase):
    """
    Apply DNN detect filter to the input.
    """
    def setDnn_backend(self, value: int) -> None:
        """
        Sets the value for dnn_backend.

        Args:
            value (int): The value to set.
        """
        ...
    def getDnn_backend(self) -> int:
        """
        Gets the value for dnn_backend.

        Returns:
            int: The current value.
        """
        ...
    def setModel(self, value: Any) -> None:
        """
        Sets the value for model.

        Args:
            value (Any): The value to set.
        """
        ...
    def getModel(self) -> Any:
        """
        Gets the value for model.

        Returns:
            Any: The current value.
        """
        ...
    def setInput(self, value: Any) -> None:
        """
        Sets the value for input.

        Args:
            value (Any): The value to set.
        """
        ...
    def getInput(self) -> Any:
        """
        Gets the value for input.

        Returns:
            Any: The current value.
        """
        ...
    def setOutput(self, value: Any) -> None:
        """
        Sets the value for output.

        Args:
            value (Any): The value to set.
        """
        ...
    def getOutput(self) -> Any:
        """
        Gets the value for output.

        Returns:
            Any: The current value.
        """
        ...
    def setBackend_configs(self, value: Any) -> None:
        """
        Sets the value for backend_configs.

        Args:
            value (Any): The value to set.
        """
        ...
    def getBackend_configs(self) -> Any:
        """
        Gets the value for backend_configs.

        Returns:
            Any: The current value.
        """
        ...
    def setAsync(self, value: bool) -> None:
        """
        Sets the value for async.

        Args:
            value (bool): The value to set.
        """
        ...
    def getAsync(self) -> bool:
        """
        Gets the value for async.

        Returns:
            bool: The current value.
        """
        ...
    def setConfidence(self, value: float) -> None:
        """
        Sets the value for confidence.

        Args:
            value (float): The value to set.
        """
        ...
    def getConfidence(self) -> float:
        """
        Gets the value for confidence.

        Returns:
            float: The current value.
        """
        ...
    def setLabels(self, value: Any) -> None:
        """
        Sets the value for labels.

        Args:
            value (Any): The value to set.
        """
        ...
    def getLabels(self) -> Any:
        """
        Gets the value for labels.

        Returns:
            Any: The current value.
        """
        ...
    def setModel_type(self, value: int) -> None:
        """
        Sets the value for model_type.

        Args:
            value (int): The value to set.
        """
        ...
    def getModel_type(self) -> int:
        """
        Gets the value for model_type.

        Returns:
            int: The current value.
        """
        ...
    def setCell_w(self, value: int) -> None:
        """
        Sets the value for cell_w.

        Args:
            value (int): The value to set.
        """
        ...
    def getCell_w(self) -> int:
        """
        Gets the value for cell_w.

        Returns:
            int: The current value.
        """
        ...
    def setCell_h(self, value: int) -> None:
        """
        Sets the value for cell_h.

        Args:
            value (int): The value to set.
        """
        ...
    def getCell_h(self) -> int:
        """
        Gets the value for cell_h.

        Returns:
            int: The current value.
        """
        ...
    def setNb_classes(self, value: int) -> None:
        """
        Sets the value for nb_classes.

        Args:
            value (int): The value to set.
        """
        ...
    def getNb_classes(self) -> int:
        """
        Gets the value for nb_classes.

        Returns:
            int: The current value.
        """
        ...
    def setAnchors(self, value: Any) -> None:
        """
        Sets the value for anchors.

        Args:
            value (Any): The value to set.
        """
        ...
    def getAnchors(self) -> Any:
        """
        Gets the value for anchors.

        Returns:
            Any: The current value.
        """
        ...

class Dnn_processing(FilterBase):
    """
    Apply DNN processing filter to the input.
    """
    def setDnn_backend(self, value: int) -> None:
        """
        Sets the value for dnn_backend.

        Args:
            value (int): The value to set.
        """
        ...
    def getDnn_backend(self) -> int:
        """
        Gets the value for dnn_backend.

        Returns:
            int: The current value.
        """
        ...
    def setModel(self, value: Any) -> None:
        """
        Sets the value for model.

        Args:
            value (Any): The value to set.
        """
        ...
    def getModel(self) -> Any:
        """
        Gets the value for model.

        Returns:
            Any: The current value.
        """
        ...
    def setInput(self, value: Any) -> None:
        """
        Sets the value for input.

        Args:
            value (Any): The value to set.
        """
        ...
    def getInput(self) -> Any:
        """
        Gets the value for input.

        Returns:
            Any: The current value.
        """
        ...
    def setOutput(self, value: Any) -> None:
        """
        Sets the value for output.

        Args:
            value (Any): The value to set.
        """
        ...
    def getOutput(self) -> Any:
        """
        Gets the value for output.

        Returns:
            Any: The current value.
        """
        ...
    def setBackend_configs(self, value: Any) -> None:
        """
        Sets the value for backend_configs.

        Args:
            value (Any): The value to set.
        """
        ...
    def getBackend_configs(self) -> Any:
        """
        Gets the value for backend_configs.

        Returns:
            Any: The current value.
        """
        ...
    def setAsync(self, value: bool) -> None:
        """
        Sets the value for async.

        Args:
            value (bool): The value to set.
        """
        ...
    def getAsync(self) -> bool:
        """
        Gets the value for async.

        Returns:
            bool: The current value.
        """
        ...

class Doubleweave(FilterBase):
    """
    Weave input video fields into double number of frames.
    """
    def setFirst_field(self, value: int) -> None:
        """
        Sets the value for first_field.

        Args:
            value (int): The value to set.
        """
        ...
    def getFirst_field(self) -> int:
        """
        Gets the value for first_field.

        Returns:
            int: The current value.
        """
        ...

class Drawbox(FilterBase):
    """
    Draw a colored box on the input video.
    """
    def setHorizontalPositionOfTheLeftBoxEdge(self, value: Any) -> None:
        """
        Sets the value for horizontalpositionoftheleftboxedge.

        Args:
            value (Any): The value to set.
        """
        ...
    def getHorizontalPositionOfTheLeftBoxEdge(self) -> Any:
        """
        Gets the value for horizontalpositionoftheleftboxedge.

        Returns:
            Any: The current value.
        """
        ...
    def setVerticalPositionOfTheTopBoxEdge(self, value: Any) -> None:
        """
        Sets the value for verticalpositionofthetopboxedge.

        Args:
            value (Any): The value to set.
        """
        ...
    def getVerticalPositionOfTheTopBoxEdge(self) -> Any:
        """
        Gets the value for verticalpositionofthetopboxedge.

        Returns:
            Any: The current value.
        """
        ...
    def setWidth(self, value: Any) -> None:
        """
        Sets the value for width.

        Args:
            value (Any): The value to set.
        """
        ...
    def getWidth(self) -> Any:
        """
        Gets the value for width.

        Returns:
            Any: The current value.
        """
        ...
    def setHeight(self, value: Any) -> None:
        """
        Sets the value for height.

        Args:
            value (Any): The value to set.
        """
        ...
    def getHeight(self) -> Any:
        """
        Gets the value for height.

        Returns:
            Any: The current value.
        """
        ...
    def setColor(self, value: Any) -> None:
        """
        Sets the value for color.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColor(self) -> Any:
        """
        Gets the value for color.

        Returns:
            Any: The current value.
        """
        ...
    def setThickness(self, value: Any) -> None:
        """
        Sets the value for thickness.

        Args:
            value (Any): The value to set.
        """
        ...
    def getThickness(self) -> Any:
        """
        Gets the value for thickness.

        Returns:
            Any: The current value.
        """
        ...
    def setReplace(self, value: bool) -> None:
        """
        Sets the value for replace.

        Args:
            value (bool): The value to set.
        """
        ...
    def getReplace(self) -> bool:
        """
        Gets the value for replace.

        Returns:
            bool: The current value.
        """
        ...
    def setBox_source(self, value: Any) -> None:
        """
        Sets the value for box_source.

        Args:
            value (Any): The value to set.
        """
        ...
    def getBox_source(self) -> Any:
        """
        Gets the value for box_source.

        Returns:
            Any: The current value.
        """
        ...

class Drawgraph(FilterBase):
    """
    Draw a graph using input video metadata.
    """
    def setM1(self, value: Any) -> None:
        """
        Sets the value for m1.

        Args:
            value (Any): The value to set.
        """
        ...
    def getM1(self) -> Any:
        """
        Gets the value for m1.

        Returns:
            Any: The current value.
        """
        ...
    def setFg1(self, value: Any) -> None:
        """
        Sets the value for fg1.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFg1(self) -> Any:
        """
        Gets the value for fg1.

        Returns:
            Any: The current value.
        """
        ...
    def setM2(self, value: Any) -> None:
        """
        Sets the value for m2.

        Args:
            value (Any): The value to set.
        """
        ...
    def getM2(self) -> Any:
        """
        Gets the value for m2.

        Returns:
            Any: The current value.
        """
        ...
    def setFg2(self, value: Any) -> None:
        """
        Sets the value for fg2.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFg2(self) -> Any:
        """
        Gets the value for fg2.

        Returns:
            Any: The current value.
        """
        ...
    def setM3(self, value: Any) -> None:
        """
        Sets the value for m3.

        Args:
            value (Any): The value to set.
        """
        ...
    def getM3(self) -> Any:
        """
        Gets the value for m3.

        Returns:
            Any: The current value.
        """
        ...
    def setFg3(self, value: Any) -> None:
        """
        Sets the value for fg3.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFg3(self) -> Any:
        """
        Gets the value for fg3.

        Returns:
            Any: The current value.
        """
        ...
    def setM4(self, value: Any) -> None:
        """
        Sets the value for m4.

        Args:
            value (Any): The value to set.
        """
        ...
    def getM4(self) -> Any:
        """
        Gets the value for m4.

        Returns:
            Any: The current value.
        """
        ...
    def setFg4(self, value: Any) -> None:
        """
        Sets the value for fg4.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFg4(self) -> Any:
        """
        Gets the value for fg4.

        Returns:
            Any: The current value.
        """
        ...
    def setBg(self, value: Any) -> None:
        """
        Sets the value for bg.

        Args:
            value (Any): The value to set.
        """
        ...
    def getBg(self) -> Any:
        """
        Gets the value for bg.

        Returns:
            Any: The current value.
        """
        ...
    def setMin(self, value: float) -> None:
        """
        Sets the value for min.

        Args:
            value (float): The value to set.
        """
        ...
    def getMin(self) -> float:
        """
        Gets the value for min.

        Returns:
            float: The current value.
        """
        ...
    def setMax(self, value: float) -> None:
        """
        Sets the value for max.

        Args:
            value (float): The value to set.
        """
        ...
    def getMax(self) -> float:
        """
        Gets the value for max.

        Returns:
            float: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setSlide(self, value: int) -> None:
        """
        Sets the value for slide.

        Args:
            value (int): The value to set.
        """
        ...
    def getSlide(self) -> int:
        """
        Gets the value for slide.

        Returns:
            int: The current value.
        """
        ...
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...

class Drawgrid(FilterBase):
    """
    Draw a colored grid on the input video.
    """
    def setHorizontalOffset(self, value: Any) -> None:
        """
        Sets the value for horizontaloffset.

        Args:
            value (Any): The value to set.
        """
        ...
    def getHorizontalOffset(self) -> Any:
        """
        Gets the value for horizontaloffset.

        Returns:
            Any: The current value.
        """
        ...
    def setVerticalOffset(self, value: Any) -> None:
        """
        Sets the value for verticaloffset.

        Args:
            value (Any): The value to set.
        """
        ...
    def getVerticalOffset(self) -> Any:
        """
        Gets the value for verticaloffset.

        Returns:
            Any: The current value.
        """
        ...
    def setWidth(self, value: Any) -> None:
        """
        Sets the value for width.

        Args:
            value (Any): The value to set.
        """
        ...
    def getWidth(self) -> Any:
        """
        Gets the value for width.

        Returns:
            Any: The current value.
        """
        ...
    def setHeight(self, value: Any) -> None:
        """
        Sets the value for height.

        Args:
            value (Any): The value to set.
        """
        ...
    def getHeight(self) -> Any:
        """
        Gets the value for height.

        Returns:
            Any: The current value.
        """
        ...
    def setColor(self, value: Any) -> None:
        """
        Sets the value for color.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColor(self) -> Any:
        """
        Gets the value for color.

        Returns:
            Any: The current value.
        """
        ...
    def setThickness(self, value: Any) -> None:
        """
        Sets the value for thickness.

        Args:
            value (Any): The value to set.
        """
        ...
    def getThickness(self) -> Any:
        """
        Gets the value for thickness.

        Returns:
            Any: The current value.
        """
        ...
    def setReplace(self, value: bool) -> None:
        """
        Sets the value for replace.

        Args:
            value (bool): The value to set.
        """
        ...
    def getReplace(self) -> bool:
        """
        Gets the value for replace.

        Returns:
            bool: The current value.
        """
        ...

class Edgedetect(FilterBase):
    """
    Detect and draw edge.
    """
    def setHigh(self, value: float) -> None:
        """
        Sets the value for high.

        Args:
            value (float): The value to set.
        """
        ...
    def getHigh(self) -> float:
        """
        Gets the value for high.

        Returns:
            float: The current value.
        """
        ...
    def setLow(self, value: float) -> None:
        """
        Sets the value for low.

        Args:
            value (float): The value to set.
        """
        ...
    def getLow(self) -> float:
        """
        Gets the value for low.

        Returns:
            float: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Elbg(FilterBase):
    """
    Apply posterize effect, using the ELBG algorithm.
    """
    def setCodebook_length(self, value: int) -> None:
        """
        Sets the value for codebook_length.

        Args:
            value (int): The value to set.
        """
        ...
    def getCodebook_length(self) -> int:
        """
        Gets the value for codebook_length.

        Returns:
            int: The current value.
        """
        ...
    def setNb_steps(self, value: int) -> None:
        """
        Sets the value for nb_steps.

        Args:
            value (int): The value to set.
        """
        ...
    def getNb_steps(self) -> int:
        """
        Gets the value for nb_steps.

        Returns:
            int: The current value.
        """
        ...
    def setSeed(self, value: int) -> None:
        """
        Sets the value for seed.

        Args:
            value (int): The value to set.
        """
        ...
    def getSeed(self) -> int:
        """
        Gets the value for seed.

        Returns:
            int: The current value.
        """
        ...
    def setPal8(self, value: bool) -> None:
        """
        Sets the value for pal8.

        Args:
            value (bool): The value to set.
        """
        ...
    def getPal8(self) -> bool:
        """
        Gets the value for pal8.

        Returns:
            bool: The current value.
        """
        ...
    def setUse_alpha(self, value: bool) -> None:
        """
        Sets the value for use_alpha.

        Args:
            value (bool): The value to set.
        """
        ...
    def getUse_alpha(self) -> bool:
        """
        Gets the value for use_alpha.

        Returns:
            bool: The current value.
        """
        ...

class Entropy(FilterBase):
    """
    Measure video frames entropy.
    """
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...

class Epx(FilterBase):
    """
    Scale the input using EPX algorithm.
    """
    def setScaleFactor(self, value: int) -> None:
        """
        Sets the value for scalefactor.

        Args:
            value (int): The value to set.
        """
        ...
    def getScaleFactor(self) -> int:
        """
        Gets the value for scalefactor.

        Returns:
            int: The current value.
        """
        ...

class Erosion(FilterBase):
    """
    Apply erosion effect.
    """
    def setCoordinates(self, value: int) -> None:
        """
        Sets the value for coordinates.

        Args:
            value (int): The value to set.
        """
        ...
    def getCoordinates(self) -> int:
        """
        Gets the value for coordinates.

        Returns:
            int: The current value.
        """
        ...
    def setThreshold0(self, value: int) -> None:
        """
        Sets the value for threshold0.

        Args:
            value (int): The value to set.
        """
        ...
    def getThreshold0(self) -> int:
        """
        Gets the value for threshold0.

        Returns:
            int: The current value.
        """
        ...
    def setThreshold1(self, value: int) -> None:
        """
        Sets the value for threshold1.

        Args:
            value (int): The value to set.
        """
        ...
    def getThreshold1(self) -> int:
        """
        Gets the value for threshold1.

        Returns:
            int: The current value.
        """
        ...
    def setThreshold2(self, value: int) -> None:
        """
        Sets the value for threshold2.

        Args:
            value (int): The value to set.
        """
        ...
    def getThreshold2(self) -> int:
        """
        Gets the value for threshold2.

        Returns:
            int: The current value.
        """
        ...
    def setThreshold3(self, value: int) -> None:
        """
        Sets the value for threshold3.

        Args:
            value (int): The value to set.
        """
        ...
    def getThreshold3(self) -> int:
        """
        Gets the value for threshold3.

        Returns:
            int: The current value.
        """
        ...

class Estdif(FilterBase):
    """
    Apply Edge Slope Tracing deinterlace.
    """
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setParity(self, value: int) -> None:
        """
        Sets the value for parity.

        Args:
            value (int): The value to set.
        """
        ...
    def getParity(self) -> int:
        """
        Gets the value for parity.

        Returns:
            int: The current value.
        """
        ...
    def setDeint(self, value: int) -> None:
        """
        Sets the value for deint.

        Args:
            value (int): The value to set.
        """
        ...
    def getDeint(self) -> int:
        """
        Gets the value for deint.

        Returns:
            int: The current value.
        """
        ...
    def setRslope(self, value: int) -> None:
        """
        Sets the value for rslope.

        Args:
            value (int): The value to set.
        """
        ...
    def getRslope(self) -> int:
        """
        Gets the value for rslope.

        Returns:
            int: The current value.
        """
        ...
    def setRedge(self, value: int) -> None:
        """
        Sets the value for redge.

        Args:
            value (int): The value to set.
        """
        ...
    def getRedge(self) -> int:
        """
        Gets the value for redge.

        Returns:
            int: The current value.
        """
        ...
    def setEcost(self, value: int) -> None:
        """
        Sets the value for ecost.

        Args:
            value (int): The value to set.
        """
        ...
    def getEcost(self) -> int:
        """
        Gets the value for ecost.

        Returns:
            int: The current value.
        """
        ...
    def setMcost(self, value: int) -> None:
        """
        Sets the value for mcost.

        Args:
            value (int): The value to set.
        """
        ...
    def getMcost(self) -> int:
        """
        Gets the value for mcost.

        Returns:
            int: The current value.
        """
        ...
    def setDcost(self, value: int) -> None:
        """
        Sets the value for dcost.

        Args:
            value (int): The value to set.
        """
        ...
    def getDcost(self) -> int:
        """
        Gets the value for dcost.

        Returns:
            int: The current value.
        """
        ...
    def setInterp(self, value: int) -> None:
        """
        Sets the value for interp.

        Args:
            value (int): The value to set.
        """
        ...
    def getInterp(self) -> int:
        """
        Gets the value for interp.

        Returns:
            int: The current value.
        """
        ...

class Exposure(FilterBase):
    """
    Adjust exposure of the video stream.
    """
    def setExposure(self, value: float) -> None:
        """
        Sets the value for exposure.

        Args:
            value (float): The value to set.
        """
        ...
    def getExposure(self) -> float:
        """
        Gets the value for exposure.

        Returns:
            float: The current value.
        """
        ...
    def setBlack(self, value: float) -> None:
        """
        Sets the value for black.

        Args:
            value (float): The value to set.
        """
        ...
    def getBlack(self) -> float:
        """
        Gets the value for black.

        Returns:
            float: The current value.
        """
        ...

class Extractplanes(FilterBase):
    """
    Extract planes as grayscale frames.
    """
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Fade(FilterBase):
    """
    Fade in/out input video.
    """
    def setType(self, value: int) -> None:
        """
        Sets the value for type.

        Args:
            value (int): The value to set.
        """
        ...
    def getType(self) -> int:
        """
        Gets the value for type.

        Returns:
            int: The current value.
        """
        ...
    def setStart_frame(self, value: int) -> None:
        """
        Sets the value for start_frame.

        Args:
            value (int): The value to set.
        """
        ...
    def getStart_frame(self) -> int:
        """
        Gets the value for start_frame.

        Returns:
            int: The current value.
        """
        ...
    def setNb_frames(self, value: int) -> None:
        """
        Sets the value for nb_frames.

        Args:
            value (int): The value to set.
        """
        ...
    def getNb_frames(self) -> int:
        """
        Gets the value for nb_frames.

        Returns:
            int: The current value.
        """
        ...
    def setAlpha(self, value: bool) -> None:
        """
        Sets the value for alpha.

        Args:
            value (bool): The value to set.
        """
        ...
    def getAlpha(self) -> bool:
        """
        Gets the value for alpha.

        Returns:
            bool: The current value.
        """
        ...
    def setStart_time(self, value: int) -> None:
        """
        Sets the value for start_time.

        Args:
            value (int): The value to set.
        """
        ...
    def getStart_time(self) -> int:
        """
        Gets the value for start_time.

        Returns:
            int: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...
    def setColor(self, value: Any) -> None:
        """
        Sets the value for color.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColor(self) -> Any:
        """
        Gets the value for color.

        Returns:
            Any: The current value.
        """
        ...

class Feedback(FilterBase):
    """
    Apply feedback video filter.
    """
    def setTopLeftCropPosition(self, value: int) -> None:
        """
        Sets the value for topleftcropposition.

        Args:
            value (int): The value to set.
        """
        ...
    def getTopLeftCropPosition(self) -> int:
        """
        Gets the value for topleftcropposition.

        Returns:
            int: The current value.
        """
        ...
    def setCropSize(self, value: int) -> None:
        """
        Sets the value for cropsize.

        Args:
            value (int): The value to set.
        """
        ...
    def getCropSize(self) -> int:
        """
        Gets the value for cropsize.

        Returns:
            int: The current value.
        """
        ...

class Fftdnoiz(FilterBase):
    """
    Denoise frames using 3D FFT.
    """
    def setSigma(self, value: float) -> None:
        """
        Sets the value for sigma.

        Args:
            value (float): The value to set.
        """
        ...
    def getSigma(self) -> float:
        """
        Gets the value for sigma.

        Returns:
            float: The current value.
        """
        ...
    def setAmount(self, value: float) -> None:
        """
        Sets the value for amount.

        Args:
            value (float): The value to set.
        """
        ...
    def getAmount(self) -> float:
        """
        Gets the value for amount.

        Returns:
            float: The current value.
        """
        ...
    def setBlock(self, value: int) -> None:
        """
        Sets the value for block.

        Args:
            value (int): The value to set.
        """
        ...
    def getBlock(self) -> int:
        """
        Gets the value for block.

        Returns:
            int: The current value.
        """
        ...
    def setOverlap(self, value: float) -> None:
        """
        Sets the value for overlap.

        Args:
            value (float): The value to set.
        """
        ...
    def getOverlap(self) -> float:
        """
        Gets the value for overlap.

        Returns:
            float: The current value.
        """
        ...
    def setMethod(self, value: int) -> None:
        """
        Sets the value for method.

        Args:
            value (int): The value to set.
        """
        ...
    def getMethod(self) -> int:
        """
        Gets the value for method.

        Returns:
            int: The current value.
        """
        ...
    def setPrev(self, value: int) -> None:
        """
        Sets the value for prev.

        Args:
            value (int): The value to set.
        """
        ...
    def getPrev(self) -> int:
        """
        Gets the value for prev.

        Returns:
            int: The current value.
        """
        ...
    def setNext(self, value: int) -> None:
        """
        Sets the value for next.

        Args:
            value (int): The value to set.
        """
        ...
    def getNext(self) -> int:
        """
        Gets the value for next.

        Returns:
            int: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setWindow(self, value: int) -> None:
        """
        Sets the value for window.

        Args:
            value (int): The value to set.
        """
        ...
    def getWindow(self) -> int:
        """
        Gets the value for window.

        Returns:
            int: The current value.
        """
        ...

class Fftfilt(FilterBase):
    """
    Apply arbitrary expressions to pixels in frequency domain.
    """
    def setDc_Y(self, value: int) -> None:
        """
        Sets the value for dc_y.

        Args:
            value (int): The value to set.
        """
        ...
    def getDc_Y(self) -> int:
        """
        Gets the value for dc_y.

        Returns:
            int: The current value.
        """
        ...
    def setDc_U(self, value: int) -> None:
        """
        Sets the value for dc_u.

        Args:
            value (int): The value to set.
        """
        ...
    def getDc_U(self) -> int:
        """
        Gets the value for dc_u.

        Returns:
            int: The current value.
        """
        ...
    def setDc_V(self, value: int) -> None:
        """
        Sets the value for dc_v.

        Args:
            value (int): The value to set.
        """
        ...
    def getDc_V(self) -> int:
        """
        Gets the value for dc_v.

        Returns:
            int: The current value.
        """
        ...
    def setWeight_Y(self, value: Any) -> None:
        """
        Sets the value for weight_y.

        Args:
            value (Any): The value to set.
        """
        ...
    def getWeight_Y(self) -> Any:
        """
        Gets the value for weight_y.

        Returns:
            Any: The current value.
        """
        ...
    def setWeight_U(self, value: Any) -> None:
        """
        Sets the value for weight_u.

        Args:
            value (Any): The value to set.
        """
        ...
    def getWeight_U(self) -> Any:
        """
        Gets the value for weight_u.

        Returns:
            Any: The current value.
        """
        ...
    def setWeight_V(self, value: Any) -> None:
        """
        Sets the value for weight_v.

        Args:
            value (Any): The value to set.
        """
        ...
    def getWeight_V(self) -> Any:
        """
        Gets the value for weight_v.

        Returns:
            Any: The current value.
        """
        ...
    def setEval(self, value: int) -> None:
        """
        Sets the value for eval.

        Args:
            value (int): The value to set.
        """
        ...
    def getEval(self) -> int:
        """
        Gets the value for eval.

        Returns:
            int: The current value.
        """
        ...

class Field(FilterBase):
    """
    Extract a field from the input video.
    """
    def setType(self, value: int) -> None:
        """
        Sets the value for type.

        Args:
            value (int): The value to set.
        """
        ...
    def getType(self) -> int:
        """
        Gets the value for type.

        Returns:
            int: The current value.
        """
        ...

class Fieldhint(FilterBase):
    """
    Field matching using hints.
    """
    def setHint(self, value: Any) -> None:
        """
        Sets the value for hint.

        Args:
            value (Any): The value to set.
        """
        ...
    def getHint(self) -> Any:
        """
        Gets the value for hint.

        Returns:
            Any: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...

class Fieldmatch(FilterBase):
    """
    Field matching for inverse telecine.
    """
    def setOrder(self, value: int) -> None:
        """
        Sets the value for order.

        Args:
            value (int): The value to set.
        """
        ...
    def getOrder(self) -> int:
        """
        Gets the value for order.

        Returns:
            int: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setPpsrc(self, value: bool) -> None:
        """
        Sets the value for ppsrc.

        Args:
            value (bool): The value to set.
        """
        ...
    def getPpsrc(self) -> bool:
        """
        Gets the value for ppsrc.

        Returns:
            bool: The current value.
        """
        ...
    def setField(self, value: int) -> None:
        """
        Sets the value for field.

        Args:
            value (int): The value to set.
        """
        ...
    def getField(self) -> int:
        """
        Gets the value for field.

        Returns:
            int: The current value.
        """
        ...
    def setMchroma(self, value: bool) -> None:
        """
        Sets the value for mchroma.

        Args:
            value (bool): The value to set.
        """
        ...
    def getMchroma(self) -> bool:
        """
        Gets the value for mchroma.

        Returns:
            bool: The current value.
        """
        ...
    def setY1(self, value: int) -> None:
        """
        Sets the value for y1.

        Args:
            value (int): The value to set.
        """
        ...
    def getY1(self) -> int:
        """
        Gets the value for y1.

        Returns:
            int: The current value.
        """
        ...
    def setScthresh(self, value: float) -> None:
        """
        Sets the value for scthresh.

        Args:
            value (float): The value to set.
        """
        ...
    def getScthresh(self) -> float:
        """
        Gets the value for scthresh.

        Returns:
            float: The current value.
        """
        ...
    def setCombmatch(self, value: int) -> None:
        """
        Sets the value for combmatch.

        Args:
            value (int): The value to set.
        """
        ...
    def getCombmatch(self) -> int:
        """
        Gets the value for combmatch.

        Returns:
            int: The current value.
        """
        ...
    def setCombdbg(self, value: int) -> None:
        """
        Sets the value for combdbg.

        Args:
            value (int): The value to set.
        """
        ...
    def getCombdbg(self) -> int:
        """
        Gets the value for combdbg.

        Returns:
            int: The current value.
        """
        ...
    def setCthresh(self, value: int) -> None:
        """
        Sets the value for cthresh.

        Args:
            value (int): The value to set.
        """
        ...
    def getCthresh(self) -> int:
        """
        Gets the value for cthresh.

        Returns:
            int: The current value.
        """
        ...
    def setChroma(self, value: bool) -> None:
        """
        Sets the value for chroma.

        Args:
            value (bool): The value to set.
        """
        ...
    def getChroma(self) -> bool:
        """
        Gets the value for chroma.

        Returns:
            bool: The current value.
        """
        ...
    def setBlockx(self, value: int) -> None:
        """
        Sets the value for blockx.

        Args:
            value (int): The value to set.
        """
        ...
    def getBlockx(self) -> int:
        """
        Gets the value for blockx.

        Returns:
            int: The current value.
        """
        ...
    def setBlocky(self, value: int) -> None:
        """
        Sets the value for blocky.

        Args:
            value (int): The value to set.
        """
        ...
    def getBlocky(self) -> int:
        """
        Gets the value for blocky.

        Returns:
            int: The current value.
        """
        ...
    def setCombpel(self, value: int) -> None:
        """
        Sets the value for combpel.

        Args:
            value (int): The value to set.
        """
        ...
    def getCombpel(self) -> int:
        """
        Gets the value for combpel.

        Returns:
            int: The current value.
        """
        ...

class Fieldorder(FilterBase):
    """
    Set the field order.
    """
    def setOrder(self, value: int) -> None:
        """
        Sets the value for order.

        Args:
            value (int): The value to set.
        """
        ...
    def getOrder(self) -> int:
        """
        Gets the value for order.

        Returns:
            int: The current value.
        """
        ...

class Fillborders(FilterBase):
    """
    Fill borders of the input video.
    """
    def setLeft(self, value: int) -> None:
        """
        Sets the value for left.

        Args:
            value (int): The value to set.
        """
        ...
    def getLeft(self) -> int:
        """
        Gets the value for left.

        Returns:
            int: The current value.
        """
        ...
    def setRight(self, value: int) -> None:
        """
        Sets the value for right.

        Args:
            value (int): The value to set.
        """
        ...
    def getRight(self) -> int:
        """
        Gets the value for right.

        Returns:
            int: The current value.
        """
        ...
    def setTop(self, value: int) -> None:
        """
        Sets the value for top.

        Args:
            value (int): The value to set.
        """
        ...
    def getTop(self) -> int:
        """
        Gets the value for top.

        Returns:
            int: The current value.
        """
        ...
    def setBottom(self, value: int) -> None:
        """
        Sets the value for bottom.

        Args:
            value (int): The value to set.
        """
        ...
    def getBottom(self) -> int:
        """
        Gets the value for bottom.

        Returns:
            int: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setColor(self, value: Any) -> None:
        """
        Sets the value for color.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColor(self) -> Any:
        """
        Gets the value for color.

        Returns:
            Any: The current value.
        """
        ...

class Floodfill(FilterBase):
    """
    Fill area with same color with another color.
    """
    def setPixelXCoordinate(self, value: int) -> None:
        """
        Sets the value for pixelxcoordinate.

        Args:
            value (int): The value to set.
        """
        ...
    def getPixelXCoordinate(self) -> int:
        """
        Gets the value for pixelxcoordinate.

        Returns:
            int: The current value.
        """
        ...
    def setPixelYCoordinate(self, value: int) -> None:
        """
        Sets the value for pixelycoordinate.

        Args:
            value (int): The value to set.
        """
        ...
    def getPixelYCoordinate(self) -> int:
        """
        Gets the value for pixelycoordinate.

        Returns:
            int: The current value.
        """
        ...
    def setS0(self, value: int) -> None:
        """
        Sets the value for s0.

        Args:
            value (int): The value to set.
        """
        ...
    def getS0(self) -> int:
        """
        Gets the value for s0.

        Returns:
            int: The current value.
        """
        ...
    def setS1(self, value: int) -> None:
        """
        Sets the value for s1.

        Args:
            value (int): The value to set.
        """
        ...
    def getS1(self) -> int:
        """
        Gets the value for s1.

        Returns:
            int: The current value.
        """
        ...
    def setS2(self, value: int) -> None:
        """
        Sets the value for s2.

        Args:
            value (int): The value to set.
        """
        ...
    def getS2(self) -> int:
        """
        Gets the value for s2.

        Returns:
            int: The current value.
        """
        ...
    def setS3(self, value: int) -> None:
        """
        Sets the value for s3.

        Args:
            value (int): The value to set.
        """
        ...
    def getS3(self) -> int:
        """
        Gets the value for s3.

        Returns:
            int: The current value.
        """
        ...
    def setD0(self, value: int) -> None:
        """
        Sets the value for d0.

        Args:
            value (int): The value to set.
        """
        ...
    def getD0(self) -> int:
        """
        Gets the value for d0.

        Returns:
            int: The current value.
        """
        ...
    def setD1(self, value: int) -> None:
        """
        Sets the value for d1.

        Args:
            value (int): The value to set.
        """
        ...
    def getD1(self) -> int:
        """
        Gets the value for d1.

        Returns:
            int: The current value.
        """
        ...
    def setD2(self, value: int) -> None:
        """
        Sets the value for d2.

        Args:
            value (int): The value to set.
        """
        ...
    def getD2(self) -> int:
        """
        Gets the value for d2.

        Returns:
            int: The current value.
        """
        ...
    def setD3(self, value: int) -> None:
        """
        Sets the value for d3.

        Args:
            value (int): The value to set.
        """
        ...
    def getD3(self) -> int:
        """
        Gets the value for d3.

        Returns:
            int: The current value.
        """
        ...

class Format(FilterBase):
    """
    Convert the input video to one of the specified pixel formats.
    """
    def setPix_fmts(self, value: List[str]) -> None:
        """
        Sets the value for pix_fmts.

        Args:
            value (List[str]): The value to set.
        """
        ...
    def getPix_fmts(self) -> List[str]:
        """
        Gets the value for pix_fmts.

        Returns:
            List[str]: The current value.
        """
        ...
    def setColor_spaces(self, value: Any) -> None:
        """
        Sets the value for color_spaces.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColor_spaces(self) -> Any:
        """
        Gets the value for color_spaces.

        Returns:
            Any: The current value.
        """
        ...
    def setColor_ranges(self, value: Any) -> None:
        """
        Sets the value for color_ranges.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColor_ranges(self) -> Any:
        """
        Gets the value for color_ranges.

        Returns:
            Any: The current value.
        """
        ...

class Fps(FilterBase):
    """
    Force constant framerate.
    """
    def setFps(self, value: Any) -> None:
        """
        Sets the value for fps.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFps(self) -> Any:
        """
        Gets the value for fps.

        Returns:
            Any: The current value.
        """
        ...
    def setStart_time(self, value: float) -> None:
        """
        Sets the value for start_time.

        Args:
            value (float): The value to set.
        """
        ...
    def getStart_time(self) -> float:
        """
        Gets the value for start_time.

        Returns:
            float: The current value.
        """
        ...
    def setRound(self, value: int) -> None:
        """
        Sets the value for round.

        Args:
            value (int): The value to set.
        """
        ...
    def getRound(self) -> int:
        """
        Gets the value for round.

        Returns:
            int: The current value.
        """
        ...
    def setEof_action(self, value: int) -> None:
        """
        Sets the value for eof_action.

        Args:
            value (int): The value to set.
        """
        ...
    def getEof_action(self) -> int:
        """
        Gets the value for eof_action.

        Returns:
            int: The current value.
        """
        ...

class Framepack(FilterBase):
    """
    Generate a frame packed stereoscopic video.
    """
    def setFormat(self, value: int) -> None:
        """
        Sets the value for format.

        Args:
            value (int): The value to set.
        """
        ...
    def getFormat(self) -> int:
        """
        Gets the value for format.

        Returns:
            int: The current value.
        """
        ...

class Framerate(FilterBase):
    """
    Upsamples or downsamples progressive source between specified frame rates.
    """
    def setFps(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for fps.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getFps(self) -> Tuple[int, int]:
        """
        Gets the value for fps.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setInterp_start(self, value: int) -> None:
        """
        Sets the value for interp_start.

        Args:
            value (int): The value to set.
        """
        ...
    def getInterp_start(self) -> int:
        """
        Gets the value for interp_start.

        Returns:
            int: The current value.
        """
        ...
    def setInterp_end(self, value: int) -> None:
        """
        Sets the value for interp_end.

        Args:
            value (int): The value to set.
        """
        ...
    def getInterp_end(self) -> int:
        """
        Gets the value for interp_end.

        Returns:
            int: The current value.
        """
        ...
    def setScene(self, value: float) -> None:
        """
        Sets the value for scene.

        Args:
            value (float): The value to set.
        """
        ...
    def getScene(self) -> float:
        """
        Gets the value for scene.

        Returns:
            float: The current value.
        """
        ...
    def setFlags(self, value: int) -> None:
        """
        Sets the value for flags.

        Args:
            value (int): The value to set.
        """
        ...
    def getFlags(self) -> int:
        """
        Gets the value for flags.

        Returns:
            int: The current value.
        """
        ...

class Framestep(FilterBase):
    """
    Select one frame every N frames.
    """
    def setStep(self, value: int) -> None:
        """
        Sets the value for step.

        Args:
            value (int): The value to set.
        """
        ...
    def getStep(self) -> int:
        """
        Gets the value for step.

        Returns:
            int: The current value.
        """
        ...

class Freezedetect(FilterBase):
    """
    Detects frozen video input.
    """
    def setNoise(self, value: float) -> None:
        """
        Sets the value for noise.

        Args:
            value (float): The value to set.
        """
        ...
    def getNoise(self) -> float:
        """
        Gets the value for noise.

        Returns:
            float: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...

class Freezeframes(FilterBase):
    """
    Freeze video frames.
    """
    def setFirst(self, value: int) -> None:
        """
        Sets the value for first.

        Args:
            value (int): The value to set.
        """
        ...
    def getFirst(self) -> int:
        """
        Gets the value for first.

        Returns:
            int: The current value.
        """
        ...
    def setLast(self, value: int) -> None:
        """
        Sets the value for last.

        Args:
            value (int): The value to set.
        """
        ...
    def getLast(self) -> int:
        """
        Gets the value for last.

        Returns:
            int: The current value.
        """
        ...
    def setReplace(self, value: int) -> None:
        """
        Sets the value for replace.

        Args:
            value (int): The value to set.
        """
        ...
    def getReplace(self) -> int:
        """
        Gets the value for replace.

        Returns:
            int: The current value.
        """
        ...

class Fsync(FilterBase):
    """
    Synchronize video frames from external source.
    """
    def setFile(self, value: Any) -> None:
        """
        Sets the value for file.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFile(self) -> Any:
        """
        Gets the value for file.

        Returns:
            Any: The current value.
        """
        ...

class Gblur(FilterBase):
    """
    Apply Gaussian Blur filter.
    """
    def setSigma(self, value: float) -> None:
        """
        Sets the value for sigma.

        Args:
            value (float): The value to set.
        """
        ...
    def getSigma(self) -> float:
        """
        Gets the value for sigma.

        Returns:
            float: The current value.
        """
        ...
    def setSteps(self, value: int) -> None:
        """
        Sets the value for steps.

        Args:
            value (int): The value to set.
        """
        ...
    def getSteps(self) -> int:
        """
        Gets the value for steps.

        Returns:
            int: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setSigmaV(self, value: float) -> None:
        """
        Sets the value for sigmav.

        Args:
            value (float): The value to set.
        """
        ...
    def getSigmaV(self) -> float:
        """
        Gets the value for sigmav.

        Returns:
            float: The current value.
        """
        ...

class Geq(FilterBase):
    """
    Apply generic equation to each pixel.
    """
    def setLum_expr(self, value: Any) -> None:
        """
        Sets the value for lum_expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getLum_expr(self) -> Any:
        """
        Gets the value for lum_expr.

        Returns:
            Any: The current value.
        """
        ...
    def setCb_expr(self, value: Any) -> None:
        """
        Sets the value for cb_expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getCb_expr(self) -> Any:
        """
        Gets the value for cb_expr.

        Returns:
            Any: The current value.
        """
        ...
    def setCr_expr(self, value: Any) -> None:
        """
        Sets the value for cr_expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getCr_expr(self) -> Any:
        """
        Gets the value for cr_expr.

        Returns:
            Any: The current value.
        """
        ...
    def setAlpha_expr(self, value: Any) -> None:
        """
        Sets the value for alpha_expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getAlpha_expr(self) -> Any:
        """
        Gets the value for alpha_expr.

        Returns:
            Any: The current value.
        """
        ...
    def setRed_expr(self, value: Any) -> None:
        """
        Sets the value for red_expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getRed_expr(self) -> Any:
        """
        Gets the value for red_expr.

        Returns:
            Any: The current value.
        """
        ...
    def setGreen_expr(self, value: Any) -> None:
        """
        Sets the value for green_expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getGreen_expr(self) -> Any:
        """
        Gets the value for green_expr.

        Returns:
            Any: The current value.
        """
        ...
    def setBlue_expr(self, value: Any) -> None:
        """
        Sets the value for blue_expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getBlue_expr(self) -> Any:
        """
        Gets the value for blue_expr.

        Returns:
            Any: The current value.
        """
        ...
    def setInterpolation(self, value: int) -> None:
        """
        Sets the value for interpolation.

        Args:
            value (int): The value to set.
        """
        ...
    def getInterpolation(self) -> int:
        """
        Gets the value for interpolation.

        Returns:
            int: The current value.
        """
        ...

class Gradfun(FilterBase):
    """
    Debands video quickly using gradients.
    """
    def setStrength(self, value: float) -> None:
        """
        Sets the value for strength.

        Args:
            value (float): The value to set.
        """
        ...
    def getStrength(self) -> float:
        """
        Gets the value for strength.

        Returns:
            float: The current value.
        """
        ...
    def setRadius(self, value: int) -> None:
        """
        Sets the value for radius.

        Args:
            value (int): The value to set.
        """
        ...
    def getRadius(self) -> int:
        """
        Gets the value for radius.

        Returns:
            int: The current value.
        """
        ...

class Graphmonitor(FilterBase):
    """
    Show various filtergraph stats.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setOpacity(self, value: float) -> None:
        """
        Sets the value for opacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getOpacity(self) -> float:
        """
        Gets the value for opacity.

        Returns:
            float: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setFlags(self, value: int) -> None:
        """
        Sets the value for flags.

        Args:
            value (int): The value to set.
        """
        ...
    def getFlags(self) -> int:
        """
        Gets the value for flags.

        Returns:
            int: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...

class Grayworld(FilterBase):
    """
    Adjust white balance using LAB gray world algorithm
    """
    pass

class Greyedge(FilterBase):
    """
    Estimates scene illumination by grey edge assumption.
    """
    def setDifford(self, value: int) -> None:
        """
        Sets the value for difford.

        Args:
            value (int): The value to set.
        """
        ...
    def getDifford(self) -> int:
        """
        Gets the value for difford.

        Returns:
            int: The current value.
        """
        ...
    def setMinknorm(self, value: int) -> None:
        """
        Sets the value for minknorm.

        Args:
            value (int): The value to set.
        """
        ...
    def getMinknorm(self) -> int:
        """
        Gets the value for minknorm.

        Returns:
            int: The current value.
        """
        ...
    def setSigma(self, value: float) -> None:
        """
        Sets the value for sigma.

        Args:
            value (float): The value to set.
        """
        ...
    def getSigma(self) -> float:
        """
        Gets the value for sigma.

        Returns:
            float: The current value.
        """
        ...

class Guided(FilterBase):
    """
    Apply Guided filter.
    """
    def setRadius(self, value: int) -> None:
        """
        Sets the value for radius.

        Args:
            value (int): The value to set.
        """
        ...
    def getRadius(self) -> int:
        """
        Gets the value for radius.

        Returns:
            int: The current value.
        """
        ...
    def setEps(self, value: float) -> None:
        """
        Sets the value for eps.

        Args:
            value (float): The value to set.
        """
        ...
    def getEps(self) -> float:
        """
        Gets the value for eps.

        Returns:
            float: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setSub(self, value: int) -> None:
        """
        Sets the value for sub.

        Args:
            value (int): The value to set.
        """
        ...
    def getSub(self) -> int:
        """
        Gets the value for sub.

        Returns:
            int: The current value.
        """
        ...
    def setGuidance(self, value: int) -> None:
        """
        Sets the value for guidance.

        Args:
            value (int): The value to set.
        """
        ...
    def getGuidance(self) -> int:
        """
        Gets the value for guidance.

        Returns:
            int: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Haldclut(FilterBase):
    """
    Adjust colors using a Hald CLUT.
    """
    def setClut(self, value: int) -> None:
        """
        Sets the value for clut.

        Args:
            value (int): The value to set.
        """
        ...
    def getClut(self) -> int:
        """
        Gets the value for clut.

        Returns:
            int: The current value.
        """
        ...
    def setInterp(self, value: int) -> None:
        """
        Sets the value for interp.

        Args:
            value (int): The value to set.
        """
        ...
    def getInterp(self) -> int:
        """
        Gets the value for interp.

        Returns:
            int: The current value.
        """
        ...

class Hflip(FilterBase):
    """
    Horizontally flip the input video.
    """
    pass

class Histogram(FilterBase):
    """
    Compute and draw a histogram.
    """
    def setLevel_height(self, value: int) -> None:
        """
        Sets the value for level_height.

        Args:
            value (int): The value to set.
        """
        ...
    def getLevel_height(self) -> int:
        """
        Gets the value for level_height.

        Returns:
            int: The current value.
        """
        ...
    def setScale_height(self, value: int) -> None:
        """
        Sets the value for scale_height.

        Args:
            value (int): The value to set.
        """
        ...
    def getScale_height(self) -> int:
        """
        Gets the value for scale_height.

        Returns:
            int: The current value.
        """
        ...
    def setDisplay_mode(self, value: int) -> None:
        """
        Sets the value for display_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getDisplay_mode(self) -> int:
        """
        Gets the value for display_mode.

        Returns:
            int: The current value.
        """
        ...
    def setLevels_mode(self, value: int) -> None:
        """
        Sets the value for levels_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getLevels_mode(self) -> int:
        """
        Gets the value for levels_mode.

        Returns:
            int: The current value.
        """
        ...
    def setComponents(self, value: int) -> None:
        """
        Sets the value for components.

        Args:
            value (int): The value to set.
        """
        ...
    def getComponents(self) -> int:
        """
        Gets the value for components.

        Returns:
            int: The current value.
        """
        ...
    def setFgopacity(self, value: float) -> None:
        """
        Sets the value for fgopacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getFgopacity(self) -> float:
        """
        Gets the value for fgopacity.

        Returns:
            float: The current value.
        """
        ...
    def setBgopacity(self, value: float) -> None:
        """
        Sets the value for bgopacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getBgopacity(self) -> float:
        """
        Gets the value for bgopacity.

        Returns:
            float: The current value.
        """
        ...
    def setColors_mode(self, value: int) -> None:
        """
        Sets the value for colors_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getColors_mode(self) -> int:
        """
        Gets the value for colors_mode.

        Returns:
            int: The current value.
        """
        ...

class Hqx(FilterBase):
    """
    Scale the input by 2, 3 or 4 using the hq*x magnification algorithm.
    """
    def setScaleFactor(self, value: int) -> None:
        """
        Sets the value for scalefactor.

        Args:
            value (int): The value to set.
        """
        ...
    def getScaleFactor(self) -> int:
        """
        Gets the value for scalefactor.

        Returns:
            int: The current value.
        """
        ...

class Hstack(FilterBase):
    """
    Stack video inputs horizontally.
    """
    def setInputs(self, value: int) -> None:
        """
        Sets the value for inputs.

        Args:
            value (int): The value to set.
        """
        ...
    def getInputs(self) -> int:
        """
        Gets the value for inputs.

        Returns:
            int: The current value.
        """
        ...
    def setShortest(self, value: bool) -> None:
        """
        Sets the value for shortest.

        Args:
            value (bool): The value to set.
        """
        ...
    def getShortest(self) -> bool:
        """
        Gets the value for shortest.

        Returns:
            bool: The current value.
        """
        ...

class Hsvhold(FilterBase):
    """
    Turns a certain HSV range into gray.
    """
    def setHue(self, value: float) -> None:
        """
        Sets the value for hue.

        Args:
            value (float): The value to set.
        """
        ...
    def getHue(self) -> float:
        """
        Gets the value for hue.

        Returns:
            float: The current value.
        """
        ...
    def setSat(self, value: float) -> None:
        """
        Sets the value for sat.

        Args:
            value (float): The value to set.
        """
        ...
    def getSat(self) -> float:
        """
        Gets the value for sat.

        Returns:
            float: The current value.
        """
        ...
    def setVal(self, value: float) -> None:
        """
        Sets the value for val.

        Args:
            value (float): The value to set.
        """
        ...
    def getVal(self) -> float:
        """
        Gets the value for val.

        Returns:
            float: The current value.
        """
        ...
    def setSimilarity(self, value: float) -> None:
        """
        Sets the value for similarity.

        Args:
            value (float): The value to set.
        """
        ...
    def getSimilarity(self) -> float:
        """
        Gets the value for similarity.

        Returns:
            float: The current value.
        """
        ...
    def setBlend(self, value: float) -> None:
        """
        Sets the value for blend.

        Args:
            value (float): The value to set.
        """
        ...
    def getBlend(self) -> float:
        """
        Gets the value for blend.

        Returns:
            float: The current value.
        """
        ...

class Hsvkey(FilterBase):
    """
    Turns a certain HSV range into transparency. Operates on YUV colors.
    """
    def setHue(self, value: float) -> None:
        """
        Sets the value for hue.

        Args:
            value (float): The value to set.
        """
        ...
    def getHue(self) -> float:
        """
        Gets the value for hue.

        Returns:
            float: The current value.
        """
        ...
    def setSat(self, value: float) -> None:
        """
        Sets the value for sat.

        Args:
            value (float): The value to set.
        """
        ...
    def getSat(self) -> float:
        """
        Gets the value for sat.

        Returns:
            float: The current value.
        """
        ...
    def setVal(self, value: float) -> None:
        """
        Sets the value for val.

        Args:
            value (float): The value to set.
        """
        ...
    def getVal(self) -> float:
        """
        Gets the value for val.

        Returns:
            float: The current value.
        """
        ...
    def setSimilarity(self, value: float) -> None:
        """
        Sets the value for similarity.

        Args:
            value (float): The value to set.
        """
        ...
    def getSimilarity(self) -> float:
        """
        Gets the value for similarity.

        Returns:
            float: The current value.
        """
        ...
    def setBlend(self, value: float) -> None:
        """
        Sets the value for blend.

        Args:
            value (float): The value to set.
        """
        ...
    def getBlend(self) -> float:
        """
        Gets the value for blend.

        Returns:
            float: The current value.
        """
        ...

class Hue(FilterBase):
    """
    Adjust the hue and saturation of the input video.
    """
    def setHueAngleDegrees(self, value: Any) -> None:
        """
        Sets the value for hueangledegrees.

        Args:
            value (Any): The value to set.
        """
        ...
    def getHueAngleDegrees(self) -> Any:
        """
        Gets the value for hueangledegrees.

        Returns:
            Any: The current value.
        """
        ...
    def setSaturation(self, value: Any) -> None:
        """
        Sets the value for saturation.

        Args:
            value (Any): The value to set.
        """
        ...
    def getSaturation(self) -> Any:
        """
        Gets the value for saturation.

        Returns:
            Any: The current value.
        """
        ...
    def setHueAngleRadians(self, value: Any) -> None:
        """
        Sets the value for hueangleradians.

        Args:
            value (Any): The value to set.
        """
        ...
    def getHueAngleRadians(self) -> Any:
        """
        Gets the value for hueangleradians.

        Returns:
            Any: The current value.
        """
        ...
    def setBrightness(self, value: Any) -> None:
        """
        Sets the value for brightness.

        Args:
            value (Any): The value to set.
        """
        ...
    def getBrightness(self) -> Any:
        """
        Gets the value for brightness.

        Returns:
            Any: The current value.
        """
        ...

class Huesaturation(FilterBase):
    """
    Apply hue-saturation-intensity adjustments.
    """
    def setHue(self, value: float) -> None:
        """
        Sets the value for hue.

        Args:
            value (float): The value to set.
        """
        ...
    def getHue(self) -> float:
        """
        Gets the value for hue.

        Returns:
            float: The current value.
        """
        ...
    def setSaturation(self, value: float) -> None:
        """
        Sets the value for saturation.

        Args:
            value (float): The value to set.
        """
        ...
    def getSaturation(self) -> float:
        """
        Gets the value for saturation.

        Returns:
            float: The current value.
        """
        ...
    def setIntensity(self, value: float) -> None:
        """
        Sets the value for intensity.

        Args:
            value (float): The value to set.
        """
        ...
    def getIntensity(self) -> float:
        """
        Gets the value for intensity.

        Returns:
            float: The current value.
        """
        ...
    def setColors(self, value: int) -> None:
        """
        Sets the value for colors.

        Args:
            value (int): The value to set.
        """
        ...
    def getColors(self) -> int:
        """
        Gets the value for colors.

        Returns:
            int: The current value.
        """
        ...
    def setStrength(self, value: float) -> None:
        """
        Sets the value for strength.

        Args:
            value (float): The value to set.
        """
        ...
    def getStrength(self) -> float:
        """
        Gets the value for strength.

        Returns:
            float: The current value.
        """
        ...
    def setRw(self, value: float) -> None:
        """
        Sets the value for rw.

        Args:
            value (float): The value to set.
        """
        ...
    def getRw(self) -> float:
        """
        Gets the value for rw.

        Returns:
            float: The current value.
        """
        ...
    def setGw(self, value: float) -> None:
        """
        Sets the value for gw.

        Args:
            value (float): The value to set.
        """
        ...
    def getGw(self) -> float:
        """
        Gets the value for gw.

        Returns:
            float: The current value.
        """
        ...
    def setBw(self, value: float) -> None:
        """
        Sets the value for bw.

        Args:
            value (float): The value to set.
        """
        ...
    def getBw(self) -> float:
        """
        Gets the value for bw.

        Returns:
            float: The current value.
        """
        ...
    def setLightness(self, value: bool) -> None:
        """
        Sets the value for lightness.

        Args:
            value (bool): The value to set.
        """
        ...
    def getLightness(self) -> bool:
        """
        Gets the value for lightness.

        Returns:
            bool: The current value.
        """
        ...

class Hwdownload(FilterBase):
    """
    Download a hardware frame to a normal frame
    """
    pass

class Hwmap(FilterBase):
    """
    Map hardware frames
    """
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setDerive_device(self, value: Any) -> None:
        """
        Sets the value for derive_device.

        Args:
            value (Any): The value to set.
        """
        ...
    def getDerive_device(self) -> Any:
        """
        Gets the value for derive_device.

        Returns:
            Any: The current value.
        """
        ...
    def setReverse(self, value: int) -> None:
        """
        Sets the value for reverse.

        Args:
            value (int): The value to set.
        """
        ...
    def getReverse(self) -> int:
        """
        Gets the value for reverse.

        Returns:
            int: The current value.
        """
        ...

class Hwupload(FilterBase):
    """
    Upload a normal frame to a hardware frame
    """
    def setDerive_device(self, value: Any) -> None:
        """
        Sets the value for derive_device.

        Args:
            value (Any): The value to set.
        """
        ...
    def getDerive_device(self) -> Any:
        """
        Gets the value for derive_device.

        Returns:
            Any: The current value.
        """
        ...

class Hwupload_cuda(FilterBase):
    """
    Upload a system memory frame to a CUDA device.
    """
    def setDevice(self, value: int) -> None:
        """
        Sets the value for device.

        Args:
            value (int): The value to set.
        """
        ...
    def getDevice(self) -> int:
        """
        Gets the value for device.

        Returns:
            int: The current value.
        """
        ...

class Hysteresis(FilterBase):
    """
    Grow first stream into second stream by connecting components.
    """
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setThreshold(self, value: int) -> None:
        """
        Sets the value for threshold.

        Args:
            value (int): The value to set.
        """
        ...
    def getThreshold(self) -> int:
        """
        Gets the value for threshold.

        Returns:
            int: The current value.
        """
        ...

class Identity(FilterBase):
    """
    Calculate the Identity between two video streams.
    """
    pass

class Idet(FilterBase):
    """
    Interlace detect Filter.
    """
    def setIntl_thres(self, value: float) -> None:
        """
        Sets the value for intl_thres.

        Args:
            value (float): The value to set.
        """
        ...
    def getIntl_thres(self) -> float:
        """
        Gets the value for intl_thres.

        Returns:
            float: The current value.
        """
        ...
    def setProg_thres(self, value: float) -> None:
        """
        Sets the value for prog_thres.

        Args:
            value (float): The value to set.
        """
        ...
    def getProg_thres(self) -> float:
        """
        Gets the value for prog_thres.

        Returns:
            float: The current value.
        """
        ...
    def setRep_thres(self, value: float) -> None:
        """
        Sets the value for rep_thres.

        Args:
            value (float): The value to set.
        """
        ...
    def getRep_thres(self) -> float:
        """
        Gets the value for rep_thres.

        Returns:
            float: The current value.
        """
        ...
    def setHalf_life(self, value: float) -> None:
        """
        Sets the value for half_life.

        Args:
            value (float): The value to set.
        """
        ...
    def getHalf_life(self) -> float:
        """
        Gets the value for half_life.

        Returns:
            float: The current value.
        """
        ...
    def setAnalyze_interlaced_flag(self, value: int) -> None:
        """
        Sets the value for analyze_interlaced_flag.

        Args:
            value (int): The value to set.
        """
        ...
    def getAnalyze_interlaced_flag(self) -> int:
        """
        Gets the value for analyze_interlaced_flag.

        Returns:
            int: The current value.
        """
        ...

class Il(FilterBase):
    """
    Deinterleave or interleave fields.
    """
    def setLuma_mode(self, value: int) -> None:
        """
        Sets the value for luma_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getLuma_mode(self) -> int:
        """
        Gets the value for luma_mode.

        Returns:
            int: The current value.
        """
        ...
    def setChroma_mode(self, value: int) -> None:
        """
        Sets the value for chroma_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getChroma_mode(self) -> int:
        """
        Gets the value for chroma_mode.

        Returns:
            int: The current value.
        """
        ...
    def setAlpha_mode(self, value: int) -> None:
        """
        Sets the value for alpha_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getAlpha_mode(self) -> int:
        """
        Gets the value for alpha_mode.

        Returns:
            int: The current value.
        """
        ...
    def setLuma_swap(self, value: bool) -> None:
        """
        Sets the value for luma_swap.

        Args:
            value (bool): The value to set.
        """
        ...
    def getLuma_swap(self) -> bool:
        """
        Gets the value for luma_swap.

        Returns:
            bool: The current value.
        """
        ...
    def setChroma_swap(self, value: bool) -> None:
        """
        Sets the value for chroma_swap.

        Args:
            value (bool): The value to set.
        """
        ...
    def getChroma_swap(self) -> bool:
        """
        Gets the value for chroma_swap.

        Returns:
            bool: The current value.
        """
        ...
    def setAlpha_swap(self, value: bool) -> None:
        """
        Sets the value for alpha_swap.

        Args:
            value (bool): The value to set.
        """
        ...
    def getAlpha_swap(self) -> bool:
        """
        Gets the value for alpha_swap.

        Returns:
            bool: The current value.
        """
        ...

class Inflate(FilterBase):
    """
    Apply inflate effect.
    """
    def setThreshold0(self, value: int) -> None:
        """
        Sets the value for threshold0.

        Args:
            value (int): The value to set.
        """
        ...
    def getThreshold0(self) -> int:
        """
        Gets the value for threshold0.

        Returns:
            int: The current value.
        """
        ...
    def setThreshold1(self, value: int) -> None:
        """
        Sets the value for threshold1.

        Args:
            value (int): The value to set.
        """
        ...
    def getThreshold1(self) -> int:
        """
        Gets the value for threshold1.

        Returns:
            int: The current value.
        """
        ...
    def setThreshold2(self, value: int) -> None:
        """
        Sets the value for threshold2.

        Args:
            value (int): The value to set.
        """
        ...
    def getThreshold2(self) -> int:
        """
        Gets the value for threshold2.

        Returns:
            int: The current value.
        """
        ...
    def setThreshold3(self, value: int) -> None:
        """
        Sets the value for threshold3.

        Args:
            value (int): The value to set.
        """
        ...
    def getThreshold3(self) -> int:
        """
        Gets the value for threshold3.

        Returns:
            int: The current value.
        """
        ...

class Interleave(FilterBase):
    """
    Temporally interleave video inputs.
    """
    def setNb_inputs(self, value: int) -> None:
        """
        Sets the value for nb_inputs.

        Args:
            value (int): The value to set.
        """
        ...
    def getNb_inputs(self) -> int:
        """
        Gets the value for nb_inputs.

        Returns:
            int: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...

class Kirsch(FilterBase):
    """
    Apply kirsch operator.
    """
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setScale(self, value: float) -> None:
        """
        Sets the value for scale.

        Args:
            value (float): The value to set.
        """
        ...
    def getScale(self) -> float:
        """
        Gets the value for scale.

        Returns:
            float: The current value.
        """
        ...
    def setDelta(self, value: float) -> None:
        """
        Sets the value for delta.

        Args:
            value (float): The value to set.
        """
        ...
    def getDelta(self) -> float:
        """
        Gets the value for delta.

        Returns:
            float: The current value.
        """
        ...

class Lagfun(FilterBase):
    """
    Slowly update darker pixels.
    """
    def setDecay(self, value: float) -> None:
        """
        Sets the value for decay.

        Args:
            value (float): The value to set.
        """
        ...
    def getDecay(self) -> float:
        """
        Gets the value for decay.

        Returns:
            float: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Latency(FilterBase):
    """
    Report video filtering latency.
    """
    pass

class Lenscorrection(FilterBase):
    """
    Rectify the image by correcting for lens distortion.
    """
    def setCx(self, value: float) -> None:
        """
        Sets the value for cx.

        Args:
            value (float): The value to set.
        """
        ...
    def getCx(self) -> float:
        """
        Gets the value for cx.

        Returns:
            float: The current value.
        """
        ...
    def setCy(self, value: float) -> None:
        """
        Sets the value for cy.

        Args:
            value (float): The value to set.
        """
        ...
    def getCy(self) -> float:
        """
        Gets the value for cy.

        Returns:
            float: The current value.
        """
        ...
    def setK1(self, value: float) -> None:
        """
        Sets the value for k1.

        Args:
            value (float): The value to set.
        """
        ...
    def getK1(self) -> float:
        """
        Gets the value for k1.

        Returns:
            float: The current value.
        """
        ...
    def setK2(self, value: float) -> None:
        """
        Sets the value for k2.

        Args:
            value (float): The value to set.
        """
        ...
    def getK2(self) -> float:
        """
        Gets the value for k2.

        Returns:
            float: The current value.
        """
        ...
    def setInterpolationType(self, value: int) -> None:
        """
        Sets the value for interpolationtype.

        Args:
            value (int): The value to set.
        """
        ...
    def getInterpolationType(self) -> int:
        """
        Gets the value for interpolationtype.

        Returns:
            int: The current value.
        """
        ...
    def setFc(self, value: Any) -> None:
        """
        Sets the value for fc.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFc(self) -> Any:
        """
        Gets the value for fc.

        Returns:
            Any: The current value.
        """
        ...

class Limitdiff(FilterBase):
    """
    Apply filtering with limiting difference.
    """
    def setThreshold(self, value: float) -> None:
        """
        Sets the value for threshold.

        Args:
            value (float): The value to set.
        """
        ...
    def getThreshold(self) -> float:
        """
        Gets the value for threshold.

        Returns:
            float: The current value.
        """
        ...
    def setElasticity(self, value: float) -> None:
        """
        Sets the value for elasticity.

        Args:
            value (float): The value to set.
        """
        ...
    def getElasticity(self) -> float:
        """
        Gets the value for elasticity.

        Returns:
            float: The current value.
        """
        ...
    def setReference(self, value: bool) -> None:
        """
        Sets the value for reference.

        Args:
            value (bool): The value to set.
        """
        ...
    def getReference(self) -> bool:
        """
        Gets the value for reference.

        Returns:
            bool: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Limiter(FilterBase):
    """
    Limit pixels components to the specified range.
    """
    def setMin(self, value: int) -> None:
        """
        Sets the value for min.

        Args:
            value (int): The value to set.
        """
        ...
    def getMin(self) -> int:
        """
        Gets the value for min.

        Returns:
            int: The current value.
        """
        ...
    def setMax(self, value: int) -> None:
        """
        Sets the value for max.

        Args:
            value (int): The value to set.
        """
        ...
    def getMax(self) -> int:
        """
        Gets the value for max.

        Returns:
            int: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Loop(FilterBase):
    """
    Loop video frames.
    """
    def setLoop(self, value: int) -> None:
        """
        Sets the value for loop.

        Args:
            value (int): The value to set.
        """
        ...
    def getLoop(self) -> int:
        """
        Gets the value for loop.

        Returns:
            int: The current value.
        """
        ...
    def setSize(self, value: int) -> None:
        """
        Sets the value for size.

        Args:
            value (int): The value to set.
        """
        ...
    def getSize(self) -> int:
        """
        Gets the value for size.

        Returns:
            int: The current value.
        """
        ...
    def setStart(self, value: int) -> None:
        """
        Sets the value for start.

        Args:
            value (int): The value to set.
        """
        ...
    def getStart(self) -> int:
        """
        Gets the value for start.

        Returns:
            int: The current value.
        """
        ...
    def setTime(self, value: int) -> None:
        """
        Sets the value for time.

        Args:
            value (int): The value to set.
        """
        ...
    def getTime(self) -> int:
        """
        Gets the value for time.

        Returns:
            int: The current value.
        """
        ...

class Lumakey(FilterBase):
    """
    Turns a certain luma into transparency.
    """
    def setThreshold(self, value: float) -> None:
        """
        Sets the value for threshold.

        Args:
            value (float): The value to set.
        """
        ...
    def getThreshold(self) -> float:
        """
        Gets the value for threshold.

        Returns:
            float: The current value.
        """
        ...
    def setTolerance(self, value: float) -> None:
        """
        Sets the value for tolerance.

        Args:
            value (float): The value to set.
        """
        ...
    def getTolerance(self) -> float:
        """
        Gets the value for tolerance.

        Returns:
            float: The current value.
        """
        ...
    def setSoftness(self, value: float) -> None:
        """
        Sets the value for softness.

        Args:
            value (float): The value to set.
        """
        ...
    def getSoftness(self) -> float:
        """
        Gets the value for softness.

        Returns:
            float: The current value.
        """
        ...

class Lut(FilterBase):
    """
    Compute and apply a lookup table to the RGB/YUV input video.
    """
    def setC0(self, value: Any) -> None:
        """
        Sets the value for c0.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC0(self) -> Any:
        """
        Gets the value for c0.

        Returns:
            Any: The current value.
        """
        ...
    def setC1(self, value: Any) -> None:
        """
        Sets the value for c1.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC1(self) -> Any:
        """
        Gets the value for c1.

        Returns:
            Any: The current value.
        """
        ...
    def setC2(self, value: Any) -> None:
        """
        Sets the value for c2.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC2(self) -> Any:
        """
        Gets the value for c2.

        Returns:
            Any: The current value.
        """
        ...
    def setC3(self, value: Any) -> None:
        """
        Sets the value for c3.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC3(self) -> Any:
        """
        Gets the value for c3.

        Returns:
            Any: The current value.
        """
        ...
    def setY(self, value: Any) -> None:
        """
        Sets the value for y.

        Args:
            value (Any): The value to set.
        """
        ...
    def getY(self) -> Any:
        """
        Gets the value for y.

        Returns:
            Any: The current value.
        """
        ...
    def setU(self, value: Any) -> None:
        """
        Sets the value for u.

        Args:
            value (Any): The value to set.
        """
        ...
    def getU(self) -> Any:
        """
        Gets the value for u.

        Returns:
            Any: The current value.
        """
        ...
    def setV(self, value: Any) -> None:
        """
        Sets the value for v.

        Args:
            value (Any): The value to set.
        """
        ...
    def getV(self) -> Any:
        """
        Gets the value for v.

        Returns:
            Any: The current value.
        """
        ...
    def setR(self, value: Any) -> None:
        """
        Sets the value for r.

        Args:
            value (Any): The value to set.
        """
        ...
    def getR(self) -> Any:
        """
        Gets the value for r.

        Returns:
            Any: The current value.
        """
        ...
    def setG(self, value: Any) -> None:
        """
        Sets the value for g.

        Args:
            value (Any): The value to set.
        """
        ...
    def getG(self) -> Any:
        """
        Gets the value for g.

        Returns:
            Any: The current value.
        """
        ...
    def setB(self, value: Any) -> None:
        """
        Sets the value for b.

        Args:
            value (Any): The value to set.
        """
        ...
    def getB(self) -> Any:
        """
        Gets the value for b.

        Returns:
            Any: The current value.
        """
        ...
    def setA(self, value: Any) -> None:
        """
        Sets the value for a.

        Args:
            value (Any): The value to set.
        """
        ...
    def getA(self) -> Any:
        """
        Gets the value for a.

        Returns:
            Any: The current value.
        """
        ...

class Lut1d(FilterBase):
    """
    Adjust colors using a 1D LUT.
    """
    def setFile(self, value: Any) -> None:
        """
        Sets the value for file.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFile(self) -> Any:
        """
        Gets the value for file.

        Returns:
            Any: The current value.
        """
        ...
    def setInterp(self, value: int) -> None:
        """
        Sets the value for interp.

        Args:
            value (int): The value to set.
        """
        ...
    def getInterp(self) -> int:
        """
        Gets the value for interp.

        Returns:
            int: The current value.
        """
        ...

class Lut2(FilterBase):
    """
    Compute and apply a lookup table from two video inputs.
    """
    def setC0(self, value: Any) -> None:
        """
        Sets the value for c0.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC0(self) -> Any:
        """
        Gets the value for c0.

        Returns:
            Any: The current value.
        """
        ...
    def setC1(self, value: Any) -> None:
        """
        Sets the value for c1.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC1(self) -> Any:
        """
        Gets the value for c1.

        Returns:
            Any: The current value.
        """
        ...
    def setC2(self, value: Any) -> None:
        """
        Sets the value for c2.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC2(self) -> Any:
        """
        Gets the value for c2.

        Returns:
            Any: The current value.
        """
        ...
    def setC3(self, value: Any) -> None:
        """
        Sets the value for c3.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC3(self) -> Any:
        """
        Gets the value for c3.

        Returns:
            Any: The current value.
        """
        ...
    def setOutputDepth(self, value: int) -> None:
        """
        Sets the value for outputdepth.

        Args:
            value (int): The value to set.
        """
        ...
    def getOutputDepth(self) -> int:
        """
        Gets the value for outputdepth.

        Returns:
            int: The current value.
        """
        ...

class Lut3d(FilterBase):
    """
    Adjust colors using a 3D LUT.
    """
    def setFile(self, value: Any) -> None:
        """
        Sets the value for file.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFile(self) -> Any:
        """
        Gets the value for file.

        Returns:
            Any: The current value.
        """
        ...
    def setClut(self, value: int) -> None:
        """
        Sets the value for clut.

        Args:
            value (int): The value to set.
        """
        ...
    def getClut(self) -> int:
        """
        Gets the value for clut.

        Returns:
            int: The current value.
        """
        ...
    def setInterp(self, value: int) -> None:
        """
        Sets the value for interp.

        Args:
            value (int): The value to set.
        """
        ...
    def getInterp(self) -> int:
        """
        Gets the value for interp.

        Returns:
            int: The current value.
        """
        ...

class Lutrgb(FilterBase):
    """
    Compute and apply a lookup table to the RGB input video.
    """
    def setC0(self, value: Any) -> None:
        """
        Sets the value for c0.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC0(self) -> Any:
        """
        Gets the value for c0.

        Returns:
            Any: The current value.
        """
        ...
    def setC1(self, value: Any) -> None:
        """
        Sets the value for c1.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC1(self) -> Any:
        """
        Gets the value for c1.

        Returns:
            Any: The current value.
        """
        ...
    def setC2(self, value: Any) -> None:
        """
        Sets the value for c2.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC2(self) -> Any:
        """
        Gets the value for c2.

        Returns:
            Any: The current value.
        """
        ...
    def setC3(self, value: Any) -> None:
        """
        Sets the value for c3.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC3(self) -> Any:
        """
        Gets the value for c3.

        Returns:
            Any: The current value.
        """
        ...
    def setY(self, value: Any) -> None:
        """
        Sets the value for y.

        Args:
            value (Any): The value to set.
        """
        ...
    def getY(self) -> Any:
        """
        Gets the value for y.

        Returns:
            Any: The current value.
        """
        ...
    def setU(self, value: Any) -> None:
        """
        Sets the value for u.

        Args:
            value (Any): The value to set.
        """
        ...
    def getU(self) -> Any:
        """
        Gets the value for u.

        Returns:
            Any: The current value.
        """
        ...
    def setV(self, value: Any) -> None:
        """
        Sets the value for v.

        Args:
            value (Any): The value to set.
        """
        ...
    def getV(self) -> Any:
        """
        Gets the value for v.

        Returns:
            Any: The current value.
        """
        ...
    def setR(self, value: Any) -> None:
        """
        Sets the value for r.

        Args:
            value (Any): The value to set.
        """
        ...
    def getR(self) -> Any:
        """
        Gets the value for r.

        Returns:
            Any: The current value.
        """
        ...
    def setG(self, value: Any) -> None:
        """
        Sets the value for g.

        Args:
            value (Any): The value to set.
        """
        ...
    def getG(self) -> Any:
        """
        Gets the value for g.

        Returns:
            Any: The current value.
        """
        ...
    def setB(self, value: Any) -> None:
        """
        Sets the value for b.

        Args:
            value (Any): The value to set.
        """
        ...
    def getB(self) -> Any:
        """
        Gets the value for b.

        Returns:
            Any: The current value.
        """
        ...
    def setA(self, value: Any) -> None:
        """
        Sets the value for a.

        Args:
            value (Any): The value to set.
        """
        ...
    def getA(self) -> Any:
        """
        Gets the value for a.

        Returns:
            Any: The current value.
        """
        ...

class Lutyuv(FilterBase):
    """
    Compute and apply a lookup table to the YUV input video.
    """
    def setC0(self, value: Any) -> None:
        """
        Sets the value for c0.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC0(self) -> Any:
        """
        Gets the value for c0.

        Returns:
            Any: The current value.
        """
        ...
    def setC1(self, value: Any) -> None:
        """
        Sets the value for c1.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC1(self) -> Any:
        """
        Gets the value for c1.

        Returns:
            Any: The current value.
        """
        ...
    def setC2(self, value: Any) -> None:
        """
        Sets the value for c2.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC2(self) -> Any:
        """
        Gets the value for c2.

        Returns:
            Any: The current value.
        """
        ...
    def setC3(self, value: Any) -> None:
        """
        Sets the value for c3.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC3(self) -> Any:
        """
        Gets the value for c3.

        Returns:
            Any: The current value.
        """
        ...
    def setY(self, value: Any) -> None:
        """
        Sets the value for y.

        Args:
            value (Any): The value to set.
        """
        ...
    def getY(self) -> Any:
        """
        Gets the value for y.

        Returns:
            Any: The current value.
        """
        ...
    def setU(self, value: Any) -> None:
        """
        Sets the value for u.

        Args:
            value (Any): The value to set.
        """
        ...
    def getU(self) -> Any:
        """
        Gets the value for u.

        Returns:
            Any: The current value.
        """
        ...
    def setV(self, value: Any) -> None:
        """
        Sets the value for v.

        Args:
            value (Any): The value to set.
        """
        ...
    def getV(self) -> Any:
        """
        Gets the value for v.

        Returns:
            Any: The current value.
        """
        ...
    def setR(self, value: Any) -> None:
        """
        Sets the value for r.

        Args:
            value (Any): The value to set.
        """
        ...
    def getR(self) -> Any:
        """
        Gets the value for r.

        Returns:
            Any: The current value.
        """
        ...
    def setG(self, value: Any) -> None:
        """
        Sets the value for g.

        Args:
            value (Any): The value to set.
        """
        ...
    def getG(self) -> Any:
        """
        Gets the value for g.

        Returns:
            Any: The current value.
        """
        ...
    def setB(self, value: Any) -> None:
        """
        Sets the value for b.

        Args:
            value (Any): The value to set.
        """
        ...
    def getB(self) -> Any:
        """
        Gets the value for b.

        Returns:
            Any: The current value.
        """
        ...
    def setA(self, value: Any) -> None:
        """
        Sets the value for a.

        Args:
            value (Any): The value to set.
        """
        ...
    def getA(self) -> Any:
        """
        Gets the value for a.

        Returns:
            Any: The current value.
        """
        ...

class Maskedclamp(FilterBase):
    """
    Clamp first stream with second stream and third stream.
    """
    def setUndershoot(self, value: int) -> None:
        """
        Sets the value for undershoot.

        Args:
            value (int): The value to set.
        """
        ...
    def getUndershoot(self) -> int:
        """
        Gets the value for undershoot.

        Returns:
            int: The current value.
        """
        ...
    def setOvershoot(self, value: int) -> None:
        """
        Sets the value for overshoot.

        Args:
            value (int): The value to set.
        """
        ...
    def getOvershoot(self) -> int:
        """
        Gets the value for overshoot.

        Returns:
            int: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Maskedmax(FilterBase):
    """
    Apply filtering with maximum difference of two streams.
    """
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Maskedmerge(FilterBase):
    """
    Merge first stream with second stream using third stream as mask.
    """
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Maskedmin(FilterBase):
    """
    Apply filtering with minimum difference of two streams.
    """
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Maskedthreshold(FilterBase):
    """
    Pick pixels comparing absolute difference of two streams with threshold.
    """
    def setThreshold(self, value: int) -> None:
        """
        Sets the value for threshold.

        Args:
            value (int): The value to set.
        """
        ...
    def getThreshold(self) -> int:
        """
        Gets the value for threshold.

        Returns:
            int: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...

class Maskfun(FilterBase):
    """
    Create Mask.
    """
    def setLow(self, value: int) -> None:
        """
        Sets the value for low.

        Args:
            value (int): The value to set.
        """
        ...
    def getLow(self) -> int:
        """
        Gets the value for low.

        Returns:
            int: The current value.
        """
        ...
    def setHigh(self, value: int) -> None:
        """
        Sets the value for high.

        Args:
            value (int): The value to set.
        """
        ...
    def getHigh(self) -> int:
        """
        Gets the value for high.

        Returns:
            int: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setFill(self, value: int) -> None:
        """
        Sets the value for fill.

        Args:
            value (int): The value to set.
        """
        ...
    def getFill(self) -> int:
        """
        Gets the value for fill.

        Returns:
            int: The current value.
        """
        ...
    def setSum(self, value: int) -> None:
        """
        Sets the value for sum.

        Args:
            value (int): The value to set.
        """
        ...
    def getSum(self) -> int:
        """
        Gets the value for sum.

        Returns:
            int: The current value.
        """
        ...

class Median(FilterBase):
    """
    Apply Median filter.
    """
    def setRadius(self, value: int) -> None:
        """
        Sets the value for radius.

        Args:
            value (int): The value to set.
        """
        ...
    def getRadius(self) -> int:
        """
        Gets the value for radius.

        Returns:
            int: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setRadiusV(self, value: int) -> None:
        """
        Sets the value for radiusv.

        Args:
            value (int): The value to set.
        """
        ...
    def getRadiusV(self) -> int:
        """
        Gets the value for radiusv.

        Returns:
            int: The current value.
        """
        ...
    def setPercentile(self, value: float) -> None:
        """
        Sets the value for percentile.

        Args:
            value (float): The value to set.
        """
        ...
    def getPercentile(self) -> float:
        """
        Gets the value for percentile.

        Returns:
            float: The current value.
        """
        ...

class Mergeplanes(FilterBase):
    """
    Merge planes.
    """
    def setFormat(self, value: Any) -> None:
        """
        Sets the value for format.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFormat(self) -> Any:
        """
        Gets the value for format.

        Returns:
            Any: The current value.
        """
        ...
    def setMap0s(self, value: int) -> None:
        """
        Sets the value for map0s.

        Args:
            value (int): The value to set.
        """
        ...
    def getMap0s(self) -> int:
        """
        Gets the value for map0s.

        Returns:
            int: The current value.
        """
        ...
    def setMap0p(self, value: int) -> None:
        """
        Sets the value for map0p.

        Args:
            value (int): The value to set.
        """
        ...
    def getMap0p(self) -> int:
        """
        Gets the value for map0p.

        Returns:
            int: The current value.
        """
        ...
    def setMap1s(self, value: int) -> None:
        """
        Sets the value for map1s.

        Args:
            value (int): The value to set.
        """
        ...
    def getMap1s(self) -> int:
        """
        Gets the value for map1s.

        Returns:
            int: The current value.
        """
        ...
    def setMap1p(self, value: int) -> None:
        """
        Sets the value for map1p.

        Args:
            value (int): The value to set.
        """
        ...
    def getMap1p(self) -> int:
        """
        Gets the value for map1p.

        Returns:
            int: The current value.
        """
        ...
    def setMap2s(self, value: int) -> None:
        """
        Sets the value for map2s.

        Args:
            value (int): The value to set.
        """
        ...
    def getMap2s(self) -> int:
        """
        Gets the value for map2s.

        Returns:
            int: The current value.
        """
        ...
    def setMap2p(self, value: int) -> None:
        """
        Sets the value for map2p.

        Args:
            value (int): The value to set.
        """
        ...
    def getMap2p(self) -> int:
        """
        Gets the value for map2p.

        Returns:
            int: The current value.
        """
        ...
    def setMap3s(self, value: int) -> None:
        """
        Sets the value for map3s.

        Args:
            value (int): The value to set.
        """
        ...
    def getMap3s(self) -> int:
        """
        Gets the value for map3s.

        Returns:
            int: The current value.
        """
        ...
    def setMap3p(self, value: int) -> None:
        """
        Sets the value for map3p.

        Args:
            value (int): The value to set.
        """
        ...
    def getMap3p(self) -> int:
        """
        Gets the value for map3p.

        Returns:
            int: The current value.
        """
        ...

class Mestimate(FilterBase):
    """
    Generate motion vectors.
    """
    def setMethod(self, value: int) -> None:
        """
        Sets the value for method.

        Args:
            value (int): The value to set.
        """
        ...
    def getMethod(self) -> int:
        """
        Gets the value for method.

        Returns:
            int: The current value.
        """
        ...
    def setMb_size(self, value: int) -> None:
        """
        Sets the value for mb_size.

        Args:
            value (int): The value to set.
        """
        ...
    def getMb_size(self) -> int:
        """
        Gets the value for mb_size.

        Returns:
            int: The current value.
        """
        ...
    def setSearch_param(self, value: int) -> None:
        """
        Sets the value for search_param.

        Args:
            value (int): The value to set.
        """
        ...
    def getSearch_param(self) -> int:
        """
        Gets the value for search_param.

        Returns:
            int: The current value.
        """
        ...

class Metadata(FilterBase):
    """
    Manipulate video frame metadata.
    """
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setKey(self, value: Any) -> None:
        """
        Sets the value for key.

        Args:
            value (Any): The value to set.
        """
        ...
    def getKey(self) -> Any:
        """
        Gets the value for key.

        Returns:
            Any: The current value.
        """
        ...
    def setValue(self, value: Any) -> None:
        """
        Sets the value for value.

        Args:
            value (Any): The value to set.
        """
        ...
    def getValue(self) -> Any:
        """
        Gets the value for value.

        Returns:
            Any: The current value.
        """
        ...
    def setFunction(self, value: int) -> None:
        """
        Sets the value for function.

        Args:
            value (int): The value to set.
        """
        ...
    def getFunction(self) -> int:
        """
        Gets the value for function.

        Returns:
            int: The current value.
        """
        ...
    def setExpr(self, value: Any) -> None:
        """
        Sets the value for expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getExpr(self) -> Any:
        """
        Gets the value for expr.

        Returns:
            Any: The current value.
        """
        ...
    def setFile(self, value: Any) -> None:
        """
        Sets the value for file.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFile(self) -> Any:
        """
        Gets the value for file.

        Returns:
            Any: The current value.
        """
        ...
    def setDirect(self, value: bool) -> None:
        """
        Sets the value for direct.

        Args:
            value (bool): The value to set.
        """
        ...
    def getDirect(self) -> bool:
        """
        Gets the value for direct.

        Returns:
            bool: The current value.
        """
        ...

class Midequalizer(FilterBase):
    """
    Apply Midway Equalization.
    """
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Minterpolate(FilterBase):
    """
    Frame rate conversion using Motion Interpolation.
    """
    def setFps(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for fps.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getFps(self) -> Tuple[int, int]:
        """
        Gets the value for fps.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setMi_mode(self, value: int) -> None:
        """
        Sets the value for mi_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMi_mode(self) -> int:
        """
        Gets the value for mi_mode.

        Returns:
            int: The current value.
        """
        ...
    def setMc_mode(self, value: int) -> None:
        """
        Sets the value for mc_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMc_mode(self) -> int:
        """
        Gets the value for mc_mode.

        Returns:
            int: The current value.
        """
        ...
    def setMe_mode(self, value: int) -> None:
        """
        Sets the value for me_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMe_mode(self) -> int:
        """
        Gets the value for me_mode.

        Returns:
            int: The current value.
        """
        ...
    def setMe(self, value: int) -> None:
        """
        Sets the value for me.

        Args:
            value (int): The value to set.
        """
        ...
    def getMe(self) -> int:
        """
        Gets the value for me.

        Returns:
            int: The current value.
        """
        ...
    def setMb_size(self, value: int) -> None:
        """
        Sets the value for mb_size.

        Args:
            value (int): The value to set.
        """
        ...
    def getMb_size(self) -> int:
        """
        Gets the value for mb_size.

        Returns:
            int: The current value.
        """
        ...
    def setSearch_param(self, value: int) -> None:
        """
        Sets the value for search_param.

        Args:
            value (int): The value to set.
        """
        ...
    def getSearch_param(self) -> int:
        """
        Gets the value for search_param.

        Returns:
            int: The current value.
        """
        ...
    def setVsbmc(self, value: int) -> None:
        """
        Sets the value for vsbmc.

        Args:
            value (int): The value to set.
        """
        ...
    def getVsbmc(self) -> int:
        """
        Gets the value for vsbmc.

        Returns:
            int: The current value.
        """
        ...
    def setScd(self, value: int) -> None:
        """
        Sets the value for scd.

        Args:
            value (int): The value to set.
        """
        ...
    def getScd(self) -> int:
        """
        Gets the value for scd.

        Returns:
            int: The current value.
        """
        ...
    def setScd_threshold(self, value: float) -> None:
        """
        Sets the value for scd_threshold.

        Args:
            value (float): The value to set.
        """
        ...
    def getScd_threshold(self) -> float:
        """
        Gets the value for scd_threshold.

        Returns:
            float: The current value.
        """
        ...

class Mix(FilterBase):
    """
    Mix video inputs.
    """
    def setInputs(self, value: int) -> None:
        """
        Sets the value for inputs.

        Args:
            value (int): The value to set.
        """
        ...
    def getInputs(self) -> int:
        """
        Gets the value for inputs.

        Returns:
            int: The current value.
        """
        ...
    def setWeights(self, value: Any) -> None:
        """
        Sets the value for weights.

        Args:
            value (Any): The value to set.
        """
        ...
    def getWeights(self) -> Any:
        """
        Gets the value for weights.

        Returns:
            Any: The current value.
        """
        ...
    def setScale(self, value: float) -> None:
        """
        Sets the value for scale.

        Args:
            value (float): The value to set.
        """
        ...
    def getScale(self) -> float:
        """
        Gets the value for scale.

        Returns:
            float: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...

class Monochrome(FilterBase):
    """
    Convert video to gray using custom color filter.
    """
    def setCb(self, value: float) -> None:
        """
        Sets the value for cb.

        Args:
            value (float): The value to set.
        """
        ...
    def getCb(self) -> float:
        """
        Gets the value for cb.

        Returns:
            float: The current value.
        """
        ...
    def setCr(self, value: float) -> None:
        """
        Sets the value for cr.

        Args:
            value (float): The value to set.
        """
        ...
    def getCr(self) -> float:
        """
        Gets the value for cr.

        Returns:
            float: The current value.
        """
        ...
    def setSize(self, value: float) -> None:
        """
        Sets the value for size.

        Args:
            value (float): The value to set.
        """
        ...
    def getSize(self) -> float:
        """
        Gets the value for size.

        Returns:
            float: The current value.
        """
        ...
    def setHigh(self, value: float) -> None:
        """
        Sets the value for high.

        Args:
            value (float): The value to set.
        """
        ...
    def getHigh(self) -> float:
        """
        Gets the value for high.

        Returns:
            float: The current value.
        """
        ...

class Morpho(FilterBase):
    """
    Apply Morphological filter.
    """
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setStructure(self, value: int) -> None:
        """
        Sets the value for structure.

        Args:
            value (int): The value to set.
        """
        ...
    def getStructure(self) -> int:
        """
        Gets the value for structure.

        Returns:
            int: The current value.
        """
        ...

class Msad(FilterBase):
    """
    Calculate the MSAD between two video streams.
    """
    pass

class Multiply(FilterBase):
    """
    Multiply first video stream with second video stream.
    """
    def setScale(self, value: float) -> None:
        """
        Sets the value for scale.

        Args:
            value (float): The value to set.
        """
        ...
    def getScale(self) -> float:
        """
        Gets the value for scale.

        Returns:
            float: The current value.
        """
        ...
    def setOffset(self, value: float) -> None:
        """
        Sets the value for offset.

        Args:
            value (float): The value to set.
        """
        ...
    def getOffset(self) -> float:
        """
        Gets the value for offset.

        Returns:
            float: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Negate(FilterBase):
    """
    Negate input video.
    """
    def setComponents(self, value: int) -> None:
        """
        Sets the value for components.

        Args:
            value (int): The value to set.
        """
        ...
    def getComponents(self) -> int:
        """
        Gets the value for components.

        Returns:
            int: The current value.
        """
        ...
    def setNegate_alpha(self, value: bool) -> None:
        """
        Sets the value for negate_alpha.

        Args:
            value (bool): The value to set.
        """
        ...
    def getNegate_alpha(self) -> bool:
        """
        Gets the value for negate_alpha.

        Returns:
            bool: The current value.
        """
        ...

class Nlmeans(FilterBase):
    """
    Non-local means denoiser.
    """
    def setDenoisingStrength(self, value: float) -> None:
        """
        Sets the value for denoisingstrength.

        Args:
            value (float): The value to set.
        """
        ...
    def getDenoisingStrength(self) -> float:
        """
        Gets the value for denoisingstrength.

        Returns:
            float: The current value.
        """
        ...
    def setPatchSize(self, value: int) -> None:
        """
        Sets the value for patchsize.

        Args:
            value (int): The value to set.
        """
        ...
    def getPatchSize(self) -> int:
        """
        Gets the value for patchsize.

        Returns:
            int: The current value.
        """
        ...
    def setPc(self, value: int) -> None:
        """
        Sets the value for pc.

        Args:
            value (int): The value to set.
        """
        ...
    def getPc(self) -> int:
        """
        Gets the value for pc.

        Returns:
            int: The current value.
        """
        ...
    def setResearchWindow(self, value: int) -> None:
        """
        Sets the value for researchwindow.

        Args:
            value (int): The value to set.
        """
        ...
    def getResearchWindow(self) -> int:
        """
        Gets the value for researchwindow.

        Returns:
            int: The current value.
        """
        ...
    def setRc(self, value: int) -> None:
        """
        Sets the value for rc.

        Args:
            value (int): The value to set.
        """
        ...
    def getRc(self) -> int:
        """
        Gets the value for rc.

        Returns:
            int: The current value.
        """
        ...

class Noformat(FilterBase):
    """
    Force libavfilter not to use any of the specified pixel formats for the input to the next filter.
    """
    def setPix_fmts(self, value: List[str]) -> None:
        """
        Sets the value for pix_fmts.

        Args:
            value (List[str]): The value to set.
        """
        ...
    def getPix_fmts(self) -> List[str]:
        """
        Gets the value for pix_fmts.

        Returns:
            List[str]: The current value.
        """
        ...
    def setColor_spaces(self, value: Any) -> None:
        """
        Sets the value for color_spaces.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColor_spaces(self) -> Any:
        """
        Gets the value for color_spaces.

        Returns:
            Any: The current value.
        """
        ...
    def setColor_ranges(self, value: Any) -> None:
        """
        Sets the value for color_ranges.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColor_ranges(self) -> Any:
        """
        Gets the value for color_ranges.

        Returns:
            Any: The current value.
        """
        ...

class Noise(FilterBase):
    """
    Add noise.
    """
    def setAll_seed(self, value: int) -> None:
        """
        Sets the value for all_seed.

        Args:
            value (int): The value to set.
        """
        ...
    def getAll_seed(self) -> int:
        """
        Gets the value for all_seed.

        Returns:
            int: The current value.
        """
        ...
    def setAll_strength(self, value: int) -> None:
        """
        Sets the value for all_strength.

        Args:
            value (int): The value to set.
        """
        ...
    def getAll_strength(self) -> int:
        """
        Gets the value for all_strength.

        Returns:
            int: The current value.
        """
        ...
    def setAll_flags(self, value: int) -> None:
        """
        Sets the value for all_flags.

        Args:
            value (int): The value to set.
        """
        ...
    def getAll_flags(self) -> int:
        """
        Gets the value for all_flags.

        Returns:
            int: The current value.
        """
        ...
    def setC0_flags(self, value: int) -> None:
        """
        Sets the value for c0_flags.

        Args:
            value (int): The value to set.
        """
        ...
    def getC0_flags(self) -> int:
        """
        Gets the value for c0_flags.

        Returns:
            int: The current value.
        """
        ...
    def setC1_seed(self, value: int) -> None:
        """
        Sets the value for c1_seed.

        Args:
            value (int): The value to set.
        """
        ...
    def getC1_seed(self) -> int:
        """
        Gets the value for c1_seed.

        Returns:
            int: The current value.
        """
        ...
    def setC1_strength(self, value: int) -> None:
        """
        Sets the value for c1_strength.

        Args:
            value (int): The value to set.
        """
        ...
    def getC1_strength(self) -> int:
        """
        Gets the value for c1_strength.

        Returns:
            int: The current value.
        """
        ...
    def setC1_flags(self, value: int) -> None:
        """
        Sets the value for c1_flags.

        Args:
            value (int): The value to set.
        """
        ...
    def getC1_flags(self) -> int:
        """
        Gets the value for c1_flags.

        Returns:
            int: The current value.
        """
        ...
    def setC2_seed(self, value: int) -> None:
        """
        Sets the value for c2_seed.

        Args:
            value (int): The value to set.
        """
        ...
    def getC2_seed(self) -> int:
        """
        Gets the value for c2_seed.

        Returns:
            int: The current value.
        """
        ...
    def setC2_strength(self, value: int) -> None:
        """
        Sets the value for c2_strength.

        Args:
            value (int): The value to set.
        """
        ...
    def getC2_strength(self) -> int:
        """
        Gets the value for c2_strength.

        Returns:
            int: The current value.
        """
        ...
    def setC2_flags(self, value: int) -> None:
        """
        Sets the value for c2_flags.

        Args:
            value (int): The value to set.
        """
        ...
    def getC2_flags(self) -> int:
        """
        Gets the value for c2_flags.

        Returns:
            int: The current value.
        """
        ...
    def setC3_seed(self, value: int) -> None:
        """
        Sets the value for c3_seed.

        Args:
            value (int): The value to set.
        """
        ...
    def getC3_seed(self) -> int:
        """
        Gets the value for c3_seed.

        Returns:
            int: The current value.
        """
        ...
    def setC3_strength(self, value: int) -> None:
        """
        Sets the value for c3_strength.

        Args:
            value (int): The value to set.
        """
        ...
    def getC3_strength(self) -> int:
        """
        Gets the value for c3_strength.

        Returns:
            int: The current value.
        """
        ...
    def setC3_flags(self, value: int) -> None:
        """
        Sets the value for c3_flags.

        Args:
            value (int): The value to set.
        """
        ...
    def getC3_flags(self) -> int:
        """
        Gets the value for c3_flags.

        Returns:
            int: The current value.
        """
        ...

class Normalize(FilterBase):
    """
    Normalize RGB video.
    """
    def setBlackpt(self, value: Any) -> None:
        """
        Sets the value for blackpt.

        Args:
            value (Any): The value to set.
        """
        ...
    def getBlackpt(self) -> Any:
        """
        Gets the value for blackpt.

        Returns:
            Any: The current value.
        """
        ...
    def setWhitept(self, value: Any) -> None:
        """
        Sets the value for whitept.

        Args:
            value (Any): The value to set.
        """
        ...
    def getWhitept(self) -> Any:
        """
        Gets the value for whitept.

        Returns:
            Any: The current value.
        """
        ...
    def setSmoothing(self, value: int) -> None:
        """
        Sets the value for smoothing.

        Args:
            value (int): The value to set.
        """
        ...
    def getSmoothing(self) -> int:
        """
        Gets the value for smoothing.

        Returns:
            int: The current value.
        """
        ...
    def setIndependence(self, value: float) -> None:
        """
        Sets the value for independence.

        Args:
            value (float): The value to set.
        """
        ...
    def getIndependence(self) -> float:
        """
        Gets the value for independence.

        Returns:
            float: The current value.
        """
        ...
    def setStrength(self, value: float) -> None:
        """
        Sets the value for strength.

        Args:
            value (float): The value to set.
        """
        ...
    def getStrength(self) -> float:
        """
        Gets the value for strength.

        Returns:
            float: The current value.
        """
        ...

class Null(FilterBase):
    """
    Pass the source unchanged to the output.
    """
    pass

class Oscilloscope(FilterBase):
    """
    2D Video Oscilloscope.
    """
    def setScopeXPosition(self, value: float) -> None:
        """
        Sets the value for scopexposition.

        Args:
            value (float): The value to set.
        """
        ...
    def getScopeXPosition(self) -> float:
        """
        Gets the value for scopexposition.

        Returns:
            float: The current value.
        """
        ...
    def setScopeYPosition(self, value: float) -> None:
        """
        Sets the value for scopeyposition.

        Args:
            value (float): The value to set.
        """
        ...
    def getScopeYPosition(self) -> float:
        """
        Gets the value for scopeyposition.

        Returns:
            float: The current value.
        """
        ...
    def setScopeSize(self, value: float) -> None:
        """
        Sets the value for scopesize.

        Args:
            value (float): The value to set.
        """
        ...
    def getScopeSize(self) -> float:
        """
        Gets the value for scopesize.

        Returns:
            float: The current value.
        """
        ...
    def setScopeTilt(self, value: float) -> None:
        """
        Sets the value for scopetilt.

        Args:
            value (float): The value to set.
        """
        ...
    def getScopeTilt(self) -> float:
        """
        Gets the value for scopetilt.

        Returns:
            float: The current value.
        """
        ...
    def setTraceOpacity(self, value: float) -> None:
        """
        Sets the value for traceopacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getTraceOpacity(self) -> float:
        """
        Gets the value for traceopacity.

        Returns:
            float: The current value.
        """
        ...
    def setTx(self, value: float) -> None:
        """
        Sets the value for tx.

        Args:
            value (float): The value to set.
        """
        ...
    def getTx(self) -> float:
        """
        Gets the value for tx.

        Returns:
            float: The current value.
        """
        ...
    def setTy(self, value: float) -> None:
        """
        Sets the value for ty.

        Args:
            value (float): The value to set.
        """
        ...
    def getTy(self) -> float:
        """
        Gets the value for ty.

        Returns:
            float: The current value.
        """
        ...
    def setTw(self, value: float) -> None:
        """
        Sets the value for tw.

        Args:
            value (float): The value to set.
        """
        ...
    def getTw(self) -> float:
        """
        Gets the value for tw.

        Returns:
            float: The current value.
        """
        ...
    def setTh(self, value: float) -> None:
        """
        Sets the value for th.

        Args:
            value (float): The value to set.
        """
        ...
    def getTh(self) -> float:
        """
        Gets the value for th.

        Returns:
            float: The current value.
        """
        ...
    def setComponentsToTrace(self, value: int) -> None:
        """
        Sets the value for componentstotrace.

        Args:
            value (int): The value to set.
        """
        ...
    def getComponentsToTrace(self) -> int:
        """
        Gets the value for componentstotrace.

        Returns:
            int: The current value.
        """
        ...
    def setDrawTraceGrid(self, value: bool) -> None:
        """
        Sets the value for drawtracegrid.

        Args:
            value (bool): The value to set.
        """
        ...
    def getDrawTraceGrid(self) -> bool:
        """
        Gets the value for drawtracegrid.

        Returns:
            bool: The current value.
        """
        ...
    def setSt(self, value: bool) -> None:
        """
        Sets the value for st.

        Args:
            value (bool): The value to set.
        """
        ...
    def getSt(self) -> bool:
        """
        Gets the value for st.

        Returns:
            bool: The current value.
        """
        ...
    def setSc(self, value: bool) -> None:
        """
        Sets the value for sc.

        Args:
            value (bool): The value to set.
        """
        ...
    def getSc(self) -> bool:
        """
        Gets the value for sc.

        Returns:
            bool: The current value.
        """
        ...

class Overlay(FilterBase):
    """
    Overlay a video source on top of the input.
    """
    def setX(self, value: Any) -> None:
        """
        Sets the value for x.

        Args:
            value (Any): The value to set.
        """
        ...
    def getX(self) -> Any:
        """
        Gets the value for x.

        Returns:
            Any: The current value.
        """
        ...
    def setY(self, value: Any) -> None:
        """
        Sets the value for y.

        Args:
            value (Any): The value to set.
        """
        ...
    def getY(self) -> Any:
        """
        Gets the value for y.

        Returns:
            Any: The current value.
        """
        ...
    def setEof_action(self, value: int) -> None:
        """
        Sets the value for eof_action.

        Args:
            value (int): The value to set.
        """
        ...
    def getEof_action(self) -> int:
        """
        Gets the value for eof_action.

        Returns:
            int: The current value.
        """
        ...
    def setEval(self, value: int) -> None:
        """
        Sets the value for eval.

        Args:
            value (int): The value to set.
        """
        ...
    def getEval(self) -> int:
        """
        Gets the value for eval.

        Returns:
            int: The current value.
        """
        ...
    def setShortest(self, value: bool) -> None:
        """
        Sets the value for shortest.

        Args:
            value (bool): The value to set.
        """
        ...
    def getShortest(self) -> bool:
        """
        Gets the value for shortest.

        Returns:
            bool: The current value.
        """
        ...
    def setFormat(self, value: int) -> None:
        """
        Sets the value for format.

        Args:
            value (int): The value to set.
        """
        ...
    def getFormat(self) -> int:
        """
        Gets the value for format.

        Returns:
            int: The current value.
        """
        ...
    def setRepeatlast(self, value: bool) -> None:
        """
        Sets the value for repeatlast.

        Args:
            value (bool): The value to set.
        """
        ...
    def getRepeatlast(self) -> bool:
        """
        Gets the value for repeatlast.

        Returns:
            bool: The current value.
        """
        ...
    def setAlpha(self, value: int) -> None:
        """
        Sets the value for alpha.

        Args:
            value (int): The value to set.
        """
        ...
    def getAlpha(self) -> int:
        """
        Gets the value for alpha.

        Returns:
            int: The current value.
        """
        ...

class Pad(FilterBase):
    """
    Pad the input video.
    """
    def setWidth(self, value: Any) -> None:
        """
        Sets the value for width.

        Args:
            value (Any): The value to set.
        """
        ...
    def getWidth(self) -> Any:
        """
        Gets the value for width.

        Returns:
            Any: The current value.
        """
        ...
    def setHeight(self, value: Any) -> None:
        """
        Sets the value for height.

        Args:
            value (Any): The value to set.
        """
        ...
    def getHeight(self) -> Any:
        """
        Gets the value for height.

        Returns:
            Any: The current value.
        """
        ...
    def setXOffsetForTheInputImagePosition(self, value: Any) -> None:
        """
        Sets the value for xoffsetfortheinputimageposition.

        Args:
            value (Any): The value to set.
        """
        ...
    def getXOffsetForTheInputImagePosition(self) -> Any:
        """
        Gets the value for xoffsetfortheinputimageposition.

        Returns:
            Any: The current value.
        """
        ...
    def setYOffsetForTheInputImagePosition(self, value: Any) -> None:
        """
        Sets the value for yoffsetfortheinputimageposition.

        Args:
            value (Any): The value to set.
        """
        ...
    def getYOffsetForTheInputImagePosition(self) -> Any:
        """
        Gets the value for yoffsetfortheinputimageposition.

        Returns:
            Any: The current value.
        """
        ...
    def setColor(self, value: Any) -> None:
        """
        Sets the value for color.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColor(self) -> Any:
        """
        Gets the value for color.

        Returns:
            Any: The current value.
        """
        ...
    def setEval(self, value: int) -> None:
        """
        Sets the value for eval.

        Args:
            value (int): The value to set.
        """
        ...
    def getEval(self) -> int:
        """
        Gets the value for eval.

        Returns:
            int: The current value.
        """
        ...
    def setAspect(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for aspect.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getAspect(self) -> Tuple[int, int]:
        """
        Gets the value for aspect.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...

class Palettegen(FilterBase):
    """
    Find the optimal palette for a given stream.
    """
    def setMax_colors(self, value: int) -> None:
        """
        Sets the value for max_colors.

        Args:
            value (int): The value to set.
        """
        ...
    def getMax_colors(self) -> int:
        """
        Gets the value for max_colors.

        Returns:
            int: The current value.
        """
        ...
    def setReserve_transparent(self, value: bool) -> None:
        """
        Sets the value for reserve_transparent.

        Args:
            value (bool): The value to set.
        """
        ...
    def getReserve_transparent(self) -> bool:
        """
        Gets the value for reserve_transparent.

        Returns:
            bool: The current value.
        """
        ...
    def setTransparency_color(self, value: Any) -> None:
        """
        Sets the value for transparency_color.

        Args:
            value (Any): The value to set.
        """
        ...
    def getTransparency_color(self) -> Any:
        """
        Gets the value for transparency_color.

        Returns:
            Any: The current value.
        """
        ...
    def setStats_mode(self, value: int) -> None:
        """
        Sets the value for stats_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getStats_mode(self) -> int:
        """
        Gets the value for stats_mode.

        Returns:
            int: The current value.
        """
        ...

class Paletteuse(FilterBase):
    """
    Use a palette to downsample an input video stream.
    """
    def setDither(self, value: int) -> None:
        """
        Sets the value for dither.

        Args:
            value (int): The value to set.
        """
        ...
    def getDither(self) -> int:
        """
        Gets the value for dither.

        Returns:
            int: The current value.
        """
        ...
    def setBayer_scale(self, value: int) -> None:
        """
        Sets the value for bayer_scale.

        Args:
            value (int): The value to set.
        """
        ...
    def getBayer_scale(self) -> int:
        """
        Gets the value for bayer_scale.

        Returns:
            int: The current value.
        """
        ...
    def setDiff_mode(self, value: int) -> None:
        """
        Sets the value for diff_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getDiff_mode(self) -> int:
        """
        Gets the value for diff_mode.

        Returns:
            int: The current value.
        """
        ...
    def setNew_(self, value: bool) -> None:
        """
        Sets the value for new_.

        Args:
            value (bool): The value to set.
        """
        ...
    def getNew_(self) -> bool:
        """
        Gets the value for new_.

        Returns:
            bool: The current value.
        """
        ...
    def setAlpha_threshold(self, value: int) -> None:
        """
        Sets the value for alpha_threshold.

        Args:
            value (int): The value to set.
        """
        ...
    def getAlpha_threshold(self) -> int:
        """
        Gets the value for alpha_threshold.

        Returns:
            int: The current value.
        """
        ...
    def setDebug_kdtree(self, value: Any) -> None:
        """
        Sets the value for debug_kdtree.

        Args:
            value (Any): The value to set.
        """
        ...
    def getDebug_kdtree(self) -> Any:
        """
        Gets the value for debug_kdtree.

        Returns:
            Any: The current value.
        """
        ...

class Photosensitivity(FilterBase):
    """
    Filter out photosensitive epilepsy seizure-inducing flashes.
    """
    def setFrames(self, value: int) -> None:
        """
        Sets the value for frames.

        Args:
            value (int): The value to set.
        """
        ...
    def getFrames(self) -> int:
        """
        Gets the value for frames.

        Returns:
            int: The current value.
        """
        ...
    def setThreshold(self, value: float) -> None:
        """
        Sets the value for threshold.

        Args:
            value (float): The value to set.
        """
        ...
    def getThreshold(self) -> float:
        """
        Gets the value for threshold.

        Returns:
            float: The current value.
        """
        ...
    def setSkip(self, value: int) -> None:
        """
        Sets the value for skip.

        Args:
            value (int): The value to set.
        """
        ...
    def getSkip(self) -> int:
        """
        Gets the value for skip.

        Returns:
            int: The current value.
        """
        ...
    def setBypass(self, value: bool) -> None:
        """
        Sets the value for bypass.

        Args:
            value (bool): The value to set.
        """
        ...
    def getBypass(self) -> bool:
        """
        Gets the value for bypass.

        Returns:
            bool: The current value.
        """
        ...

class Pixdesctest(FilterBase):
    """
    Test pixel format definitions.
    """
    pass

class Pixelize(FilterBase):
    """
    Pixelize video.
    """
    def setWidth(self, value: int) -> None:
        """
        Sets the value for width.

        Args:
            value (int): The value to set.
        """
        ...
    def getWidth(self) -> int:
        """
        Gets the value for width.

        Returns:
            int: The current value.
        """
        ...
    def setHeight(self, value: int) -> None:
        """
        Sets the value for height.

        Args:
            value (int): The value to set.
        """
        ...
    def getHeight(self) -> int:
        """
        Gets the value for height.

        Returns:
            int: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Pixscope(FilterBase):
    """
    Pixel data analysis.
    """
    def setScopeXOffset(self, value: float) -> None:
        """
        Sets the value for scopexoffset.

        Args:
            value (float): The value to set.
        """
        ...
    def getScopeXOffset(self) -> float:
        """
        Gets the value for scopexoffset.

        Returns:
            float: The current value.
        """
        ...
    def setScopeYOffset(self, value: float) -> None:
        """
        Sets the value for scopeyoffset.

        Args:
            value (float): The value to set.
        """
        ...
    def getScopeYOffset(self) -> float:
        """
        Gets the value for scopeyoffset.

        Returns:
            float: The current value.
        """
        ...
    def setScopeWidth(self, value: int) -> None:
        """
        Sets the value for scopewidth.

        Args:
            value (int): The value to set.
        """
        ...
    def getScopeWidth(self) -> int:
        """
        Gets the value for scopewidth.

        Returns:
            int: The current value.
        """
        ...
    def setScopeHeight(self, value: int) -> None:
        """
        Sets the value for scopeheight.

        Args:
            value (int): The value to set.
        """
        ...
    def getScopeHeight(self) -> int:
        """
        Gets the value for scopeheight.

        Returns:
            int: The current value.
        """
        ...
    def setWindowOpacity(self, value: float) -> None:
        """
        Sets the value for windowopacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getWindowOpacity(self) -> float:
        """
        Gets the value for windowopacity.

        Returns:
            float: The current value.
        """
        ...
    def setWx(self, value: float) -> None:
        """
        Sets the value for wx.

        Args:
            value (float): The value to set.
        """
        ...
    def getWx(self) -> float:
        """
        Gets the value for wx.

        Returns:
            float: The current value.
        """
        ...
    def setWy(self, value: float) -> None:
        """
        Sets the value for wy.

        Args:
            value (float): The value to set.
        """
        ...
    def getWy(self) -> float:
        """
        Gets the value for wy.

        Returns:
            float: The current value.
        """
        ...

class Premultiply(FilterBase):
    """
    PreMultiply first stream with first plane of second stream.
    """
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setInplace(self, value: bool) -> None:
        """
        Sets the value for inplace.

        Args:
            value (bool): The value to set.
        """
        ...
    def getInplace(self) -> bool:
        """
        Gets the value for inplace.

        Returns:
            bool: The current value.
        """
        ...

class Prewitt(FilterBase):
    """
    Apply prewitt operator.
    """
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setScale(self, value: float) -> None:
        """
        Sets the value for scale.

        Args:
            value (float): The value to set.
        """
        ...
    def getScale(self) -> float:
        """
        Gets the value for scale.

        Returns:
            float: The current value.
        """
        ...
    def setDelta(self, value: float) -> None:
        """
        Sets the value for delta.

        Args:
            value (float): The value to set.
        """
        ...
    def getDelta(self) -> float:
        """
        Gets the value for delta.

        Returns:
            float: The current value.
        """
        ...

class Pseudocolor(FilterBase):
    """
    Make pseudocolored video frames.
    """
    def setC0(self, value: Any) -> None:
        """
        Sets the value for c0.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC0(self) -> Any:
        """
        Gets the value for c0.

        Returns:
            Any: The current value.
        """
        ...
    def setC1(self, value: Any) -> None:
        """
        Sets the value for c1.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC1(self) -> Any:
        """
        Gets the value for c1.

        Returns:
            Any: The current value.
        """
        ...
    def setC2(self, value: Any) -> None:
        """
        Sets the value for c2.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC2(self) -> Any:
        """
        Gets the value for c2.

        Returns:
            Any: The current value.
        """
        ...
    def setC3(self, value: Any) -> None:
        """
        Sets the value for c3.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC3(self) -> Any:
        """
        Gets the value for c3.

        Returns:
            Any: The current value.
        """
        ...
    def setIndex(self, value: int) -> None:
        """
        Sets the value for index.

        Args:
            value (int): The value to set.
        """
        ...
    def getIndex(self) -> int:
        """
        Gets the value for index.

        Returns:
            int: The current value.
        """
        ...
    def setPreset(self, value: int) -> None:
        """
        Sets the value for preset.

        Args:
            value (int): The value to set.
        """
        ...
    def getPreset(self) -> int:
        """
        Gets the value for preset.

        Returns:
            int: The current value.
        """
        ...
    def setOpacity(self, value: float) -> None:
        """
        Sets the value for opacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getOpacity(self) -> float:
        """
        Gets the value for opacity.

        Returns:
            float: The current value.
        """
        ...

class Psnr(FilterBase):
    """
    Calculate the PSNR between two video streams.
    """
    def setStats_file(self, value: Any) -> None:
        """
        Sets the value for stats_file.

        Args:
            value (Any): The value to set.
        """
        ...
    def getStats_file(self) -> Any:
        """
        Gets the value for stats_file.

        Returns:
            Any: The current value.
        """
        ...
    def setStats_version(self, value: int) -> None:
        """
        Sets the value for stats_version.

        Args:
            value (int): The value to set.
        """
        ...
    def getStats_version(self) -> int:
        """
        Gets the value for stats_version.

        Returns:
            int: The current value.
        """
        ...
    def setOutput_max(self, value: bool) -> None:
        """
        Sets the value for output_max.

        Args:
            value (bool): The value to set.
        """
        ...
    def getOutput_max(self) -> bool:
        """
        Gets the value for output_max.

        Returns:
            bool: The current value.
        """
        ...

class Qp(FilterBase):
    """
    Change video quantization parameters.
    """
    def setQp(self, value: Any) -> None:
        """
        Sets the value for qp.

        Args:
            value (Any): The value to set.
        """
        ...
    def getQp(self) -> Any:
        """
        Gets the value for qp.

        Returns:
            Any: The current value.
        """
        ...

class Random(FilterBase):
    """
    Return random frames.
    """
    def setFrames(self, value: int) -> None:
        """
        Sets the value for frames.

        Args:
            value (int): The value to set.
        """
        ...
    def getFrames(self) -> int:
        """
        Gets the value for frames.

        Returns:
            int: The current value.
        """
        ...
    def setSeed(self, value: int) -> None:
        """
        Sets the value for seed.

        Args:
            value (int): The value to set.
        """
        ...
    def getSeed(self) -> int:
        """
        Gets the value for seed.

        Returns:
            int: The current value.
        """
        ...

class Readeia608(FilterBase):
    """
    Read EIA-608 Closed Caption codes from input video and write them to frame metadata.
    """
    def setScan_min(self, value: int) -> None:
        """
        Sets the value for scan_min.

        Args:
            value (int): The value to set.
        """
        ...
    def getScan_min(self) -> int:
        """
        Gets the value for scan_min.

        Returns:
            int: The current value.
        """
        ...
    def setScan_max(self, value: int) -> None:
        """
        Sets the value for scan_max.

        Args:
            value (int): The value to set.
        """
        ...
    def getScan_max(self) -> int:
        """
        Gets the value for scan_max.

        Returns:
            int: The current value.
        """
        ...
    def setSpw(self, value: float) -> None:
        """
        Sets the value for spw.

        Args:
            value (float): The value to set.
        """
        ...
    def getSpw(self) -> float:
        """
        Gets the value for spw.

        Returns:
            float: The current value.
        """
        ...
    def setChp(self, value: bool) -> None:
        """
        Sets the value for chp.

        Args:
            value (bool): The value to set.
        """
        ...
    def getChp(self) -> bool:
        """
        Gets the value for chp.

        Returns:
            bool: The current value.
        """
        ...
    def setLp(self, value: bool) -> None:
        """
        Sets the value for lp.

        Args:
            value (bool): The value to set.
        """
        ...
    def getLp(self) -> bool:
        """
        Gets the value for lp.

        Returns:
            bool: The current value.
        """
        ...

class Readvitc(FilterBase):
    """
    Read vertical interval timecode and write it to frame metadata.
    """
    def setScan_max(self, value: int) -> None:
        """
        Sets the value for scan_max.

        Args:
            value (int): The value to set.
        """
        ...
    def getScan_max(self) -> int:
        """
        Gets the value for scan_max.

        Returns:
            int: The current value.
        """
        ...
    def setThr_b(self, value: float) -> None:
        """
        Sets the value for thr_b.

        Args:
            value (float): The value to set.
        """
        ...
    def getThr_b(self) -> float:
        """
        Gets the value for thr_b.

        Returns:
            float: The current value.
        """
        ...
    def setThr_w(self, value: float) -> None:
        """
        Sets the value for thr_w.

        Args:
            value (float): The value to set.
        """
        ...
    def getThr_w(self) -> float:
        """
        Gets the value for thr_w.

        Returns:
            float: The current value.
        """
        ...

class Remap(FilterBase):
    """
    Remap pixels.
    """
    def setFormat(self, value: int) -> None:
        """
        Sets the value for format.

        Args:
            value (int): The value to set.
        """
        ...
    def getFormat(self) -> int:
        """
        Gets the value for format.

        Returns:
            int: The current value.
        """
        ...
    def setFill(self, value: Any) -> None:
        """
        Sets the value for fill.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFill(self) -> Any:
        """
        Gets the value for fill.

        Returns:
            Any: The current value.
        """
        ...

class Removegrain(FilterBase):
    """
    Remove grain.
    """
    def setM0(self, value: int) -> None:
        """
        Sets the value for m0.

        Args:
            value (int): The value to set.
        """
        ...
    def getM0(self) -> int:
        """
        Gets the value for m0.

        Returns:
            int: The current value.
        """
        ...
    def setM1(self, value: int) -> None:
        """
        Sets the value for m1.

        Args:
            value (int): The value to set.
        """
        ...
    def getM1(self) -> int:
        """
        Gets the value for m1.

        Returns:
            int: The current value.
        """
        ...
    def setM2(self, value: int) -> None:
        """
        Sets the value for m2.

        Args:
            value (int): The value to set.
        """
        ...
    def getM2(self) -> int:
        """
        Gets the value for m2.

        Returns:
            int: The current value.
        """
        ...
    def setM3(self, value: int) -> None:
        """
        Sets the value for m3.

        Args:
            value (int): The value to set.
        """
        ...
    def getM3(self) -> int:
        """
        Gets the value for m3.

        Returns:
            int: The current value.
        """
        ...

class Removelogo(FilterBase):
    """
    Remove a TV logo based on a mask image.
    """
    def setFilename(self, value: Any) -> None:
        """
        Sets the value for filename.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFilename(self) -> Any:
        """
        Gets the value for filename.

        Returns:
            Any: The current value.
        """
        ...

class Reverse(FilterBase):
    """
    Reverse a clip.
    """
    pass

class Rgbashift(FilterBase):
    """
    Shift RGBA.
    """
    def setRh(self, value: int) -> None:
        """
        Sets the value for rh.

        Args:
            value (int): The value to set.
        """
        ...
    def getRh(self) -> int:
        """
        Gets the value for rh.

        Returns:
            int: The current value.
        """
        ...
    def setRv(self, value: int) -> None:
        """
        Sets the value for rv.

        Args:
            value (int): The value to set.
        """
        ...
    def getRv(self) -> int:
        """
        Gets the value for rv.

        Returns:
            int: The current value.
        """
        ...
    def setGh(self, value: int) -> None:
        """
        Sets the value for gh.

        Args:
            value (int): The value to set.
        """
        ...
    def getGh(self) -> int:
        """
        Gets the value for gh.

        Returns:
            int: The current value.
        """
        ...
    def setGv(self, value: int) -> None:
        """
        Sets the value for gv.

        Args:
            value (int): The value to set.
        """
        ...
    def getGv(self) -> int:
        """
        Gets the value for gv.

        Returns:
            int: The current value.
        """
        ...
    def setBh(self, value: int) -> None:
        """
        Sets the value for bh.

        Args:
            value (int): The value to set.
        """
        ...
    def getBh(self) -> int:
        """
        Gets the value for bh.

        Returns:
            int: The current value.
        """
        ...
    def setBv(self, value: int) -> None:
        """
        Sets the value for bv.

        Args:
            value (int): The value to set.
        """
        ...
    def getBv(self) -> int:
        """
        Gets the value for bv.

        Returns:
            int: The current value.
        """
        ...
    def setAh(self, value: int) -> None:
        """
        Sets the value for ah.

        Args:
            value (int): The value to set.
        """
        ...
    def getAh(self) -> int:
        """
        Gets the value for ah.

        Returns:
            int: The current value.
        """
        ...
    def setAv(self, value: int) -> None:
        """
        Sets the value for av.

        Args:
            value (int): The value to set.
        """
        ...
    def getAv(self) -> int:
        """
        Gets the value for av.

        Returns:
            int: The current value.
        """
        ...
    def setEdge(self, value: int) -> None:
        """
        Sets the value for edge.

        Args:
            value (int): The value to set.
        """
        ...
    def getEdge(self) -> int:
        """
        Gets the value for edge.

        Returns:
            int: The current value.
        """
        ...

class Roberts(FilterBase):
    """
    Apply roberts cross operator.
    """
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setScale(self, value: float) -> None:
        """
        Sets the value for scale.

        Args:
            value (float): The value to set.
        """
        ...
    def getScale(self) -> float:
        """
        Gets the value for scale.

        Returns:
            float: The current value.
        """
        ...
    def setDelta(self, value: float) -> None:
        """
        Sets the value for delta.

        Args:
            value (float): The value to set.
        """
        ...
    def getDelta(self) -> float:
        """
        Gets the value for delta.

        Returns:
            float: The current value.
        """
        ...

class Rotate(FilterBase):
    """
    Rotate the input image.
    """
    def setAngle(self, value: Any) -> None:
        """
        Sets the value for angle.

        Args:
            value (Any): The value to set.
        """
        ...
    def getAngle(self) -> Any:
        """
        Gets the value for angle.

        Returns:
            Any: The current value.
        """
        ...
    def setOut_w(self, value: Any) -> None:
        """
        Sets the value for out_w.

        Args:
            value (Any): The value to set.
        """
        ...
    def getOut_w(self) -> Any:
        """
        Gets the value for out_w.

        Returns:
            Any: The current value.
        """
        ...
    def setOut_h(self, value: Any) -> None:
        """
        Sets the value for out_h.

        Args:
            value (Any): The value to set.
        """
        ...
    def getOut_h(self) -> Any:
        """
        Gets the value for out_h.

        Returns:
            Any: The current value.
        """
        ...
    def setFillcolor(self, value: Any) -> None:
        """
        Sets the value for fillcolor.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFillcolor(self) -> Any:
        """
        Gets the value for fillcolor.

        Returns:
            Any: The current value.
        """
        ...
    def setBilinear(self, value: bool) -> None:
        """
        Sets the value for bilinear.

        Args:
            value (bool): The value to set.
        """
        ...
    def getBilinear(self) -> bool:
        """
        Gets the value for bilinear.

        Returns:
            bool: The current value.
        """
        ...

class Scale(FilterBase):
    """
    Scale the input video size and/or convert the image format.
    """
    def setWidth(self, value: Any) -> None:
        """
        Sets the value for width.

        Args:
            value (Any): The value to set.
        """
        ...
    def getWidth(self) -> Any:
        """
        Gets the value for width.

        Returns:
            Any: The current value.
        """
        ...
    def setHeight(self, value: Any) -> None:
        """
        Sets the value for height.

        Args:
            value (Any): The value to set.
        """
        ...
    def getHeight(self) -> Any:
        """
        Gets the value for height.

        Returns:
            Any: The current value.
        """
        ...
    def setFlags(self, value: Any) -> None:
        """
        Sets the value for flags.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFlags(self) -> Any:
        """
        Gets the value for flags.

        Returns:
            Any: The current value.
        """
        ...
    def setInterl(self, value: bool) -> None:
        """
        Sets the value for interl.

        Args:
            value (bool): The value to set.
        """
        ...
    def getInterl(self) -> bool:
        """
        Gets the value for interl.

        Returns:
            bool: The current value.
        """
        ...
    def setSize(self, value: Any) -> None:
        """
        Sets the value for size.

        Args:
            value (Any): The value to set.
        """
        ...
    def getSize(self) -> Any:
        """
        Gets the value for size.

        Returns:
            Any: The current value.
        """
        ...
    def setIn_color_matrix(self, value: int) -> None:
        """
        Sets the value for in_color_matrix.

        Args:
            value (int): The value to set.
        """
        ...
    def getIn_color_matrix(self) -> int:
        """
        Gets the value for in_color_matrix.

        Returns:
            int: The current value.
        """
        ...
    def setOut_color_matrix(self, value: int) -> None:
        """
        Sets the value for out_color_matrix.

        Args:
            value (int): The value to set.
        """
        ...
    def getOut_color_matrix(self) -> int:
        """
        Gets the value for out_color_matrix.

        Returns:
            int: The current value.
        """
        ...
    def setIn_range(self, value: int) -> None:
        """
        Sets the value for in_range.

        Args:
            value (int): The value to set.
        """
        ...
    def getIn_range(self) -> int:
        """
        Gets the value for in_range.

        Returns:
            int: The current value.
        """
        ...
    def setOut_range(self, value: int) -> None:
        """
        Sets the value for out_range.

        Args:
            value (int): The value to set.
        """
        ...
    def getOut_range(self) -> int:
        """
        Gets the value for out_range.

        Returns:
            int: The current value.
        """
        ...
    def setIn_v_chr_pos(self, value: int) -> None:
        """
        Sets the value for in_v_chr_pos.

        Args:
            value (int): The value to set.
        """
        ...
    def getIn_v_chr_pos(self) -> int:
        """
        Gets the value for in_v_chr_pos.

        Returns:
            int: The current value.
        """
        ...
    def setIn_h_chr_pos(self, value: int) -> None:
        """
        Sets the value for in_h_chr_pos.

        Args:
            value (int): The value to set.
        """
        ...
    def getIn_h_chr_pos(self) -> int:
        """
        Gets the value for in_h_chr_pos.

        Returns:
            int: The current value.
        """
        ...
    def setOut_v_chr_pos(self, value: int) -> None:
        """
        Sets the value for out_v_chr_pos.

        Args:
            value (int): The value to set.
        """
        ...
    def getOut_v_chr_pos(self) -> int:
        """
        Gets the value for out_v_chr_pos.

        Returns:
            int: The current value.
        """
        ...
    def setOut_h_chr_pos(self, value: int) -> None:
        """
        Sets the value for out_h_chr_pos.

        Args:
            value (int): The value to set.
        """
        ...
    def getOut_h_chr_pos(self) -> int:
        """
        Gets the value for out_h_chr_pos.

        Returns:
            int: The current value.
        """
        ...
    def setForce_original_aspect_ratio(self, value: int) -> None:
        """
        Sets the value for force_original_aspect_ratio.

        Args:
            value (int): The value to set.
        """
        ...
    def getForce_original_aspect_ratio(self) -> int:
        """
        Gets the value for force_original_aspect_ratio.

        Returns:
            int: The current value.
        """
        ...
    def setForce_divisible_by(self, value: int) -> None:
        """
        Sets the value for force_divisible_by.

        Args:
            value (int): The value to set.
        """
        ...
    def getForce_divisible_by(self) -> int:
        """
        Gets the value for force_divisible_by.

        Returns:
            int: The current value.
        """
        ...
    def setParam0(self, value: float) -> None:
        """
        Sets the value for param0.

        Args:
            value (float): The value to set.
        """
        ...
    def getParam0(self) -> float:
        """
        Gets the value for param0.

        Returns:
            float: The current value.
        """
        ...
    def setParam1(self, value: float) -> None:
        """
        Sets the value for param1.

        Args:
            value (float): The value to set.
        """
        ...
    def getParam1(self) -> float:
        """
        Gets the value for param1.

        Returns:
            float: The current value.
        """
        ...
    def setEval(self, value: int) -> None:
        """
        Sets the value for eval.

        Args:
            value (int): The value to set.
        """
        ...
    def getEval(self) -> int:
        """
        Gets the value for eval.

        Returns:
            int: The current value.
        """
        ...

class Scale2ref(FilterBase):
    """
    Scale the input video size and/or convert the image format to the given reference.
    """
    def setWidth(self, value: Any) -> None:
        """
        Sets the value for width.

        Args:
            value (Any): The value to set.
        """
        ...
    def getWidth(self) -> Any:
        """
        Gets the value for width.

        Returns:
            Any: The current value.
        """
        ...
    def setHeight(self, value: Any) -> None:
        """
        Sets the value for height.

        Args:
            value (Any): The value to set.
        """
        ...
    def getHeight(self) -> Any:
        """
        Gets the value for height.

        Returns:
            Any: The current value.
        """
        ...
    def setFlags(self, value: Any) -> None:
        """
        Sets the value for flags.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFlags(self) -> Any:
        """
        Gets the value for flags.

        Returns:
            Any: The current value.
        """
        ...
    def setInterl(self, value: bool) -> None:
        """
        Sets the value for interl.

        Args:
            value (bool): The value to set.
        """
        ...
    def getInterl(self) -> bool:
        """
        Gets the value for interl.

        Returns:
            bool: The current value.
        """
        ...
    def setSize(self, value: Any) -> None:
        """
        Sets the value for size.

        Args:
            value (Any): The value to set.
        """
        ...
    def getSize(self) -> Any:
        """
        Gets the value for size.

        Returns:
            Any: The current value.
        """
        ...
    def setIn_color_matrix(self, value: int) -> None:
        """
        Sets the value for in_color_matrix.

        Args:
            value (int): The value to set.
        """
        ...
    def getIn_color_matrix(self) -> int:
        """
        Gets the value for in_color_matrix.

        Returns:
            int: The current value.
        """
        ...
    def setOut_color_matrix(self, value: int) -> None:
        """
        Sets the value for out_color_matrix.

        Args:
            value (int): The value to set.
        """
        ...
    def getOut_color_matrix(self) -> int:
        """
        Gets the value for out_color_matrix.

        Returns:
            int: The current value.
        """
        ...
    def setIn_range(self, value: int) -> None:
        """
        Sets the value for in_range.

        Args:
            value (int): The value to set.
        """
        ...
    def getIn_range(self) -> int:
        """
        Gets the value for in_range.

        Returns:
            int: The current value.
        """
        ...
    def setOut_range(self, value: int) -> None:
        """
        Sets the value for out_range.

        Args:
            value (int): The value to set.
        """
        ...
    def getOut_range(self) -> int:
        """
        Gets the value for out_range.

        Returns:
            int: The current value.
        """
        ...
    def setIn_v_chr_pos(self, value: int) -> None:
        """
        Sets the value for in_v_chr_pos.

        Args:
            value (int): The value to set.
        """
        ...
    def getIn_v_chr_pos(self) -> int:
        """
        Gets the value for in_v_chr_pos.

        Returns:
            int: The current value.
        """
        ...
    def setIn_h_chr_pos(self, value: int) -> None:
        """
        Sets the value for in_h_chr_pos.

        Args:
            value (int): The value to set.
        """
        ...
    def getIn_h_chr_pos(self) -> int:
        """
        Gets the value for in_h_chr_pos.

        Returns:
            int: The current value.
        """
        ...
    def setOut_v_chr_pos(self, value: int) -> None:
        """
        Sets the value for out_v_chr_pos.

        Args:
            value (int): The value to set.
        """
        ...
    def getOut_v_chr_pos(self) -> int:
        """
        Gets the value for out_v_chr_pos.

        Returns:
            int: The current value.
        """
        ...
    def setOut_h_chr_pos(self, value: int) -> None:
        """
        Sets the value for out_h_chr_pos.

        Args:
            value (int): The value to set.
        """
        ...
    def getOut_h_chr_pos(self) -> int:
        """
        Gets the value for out_h_chr_pos.

        Returns:
            int: The current value.
        """
        ...
    def setForce_original_aspect_ratio(self, value: int) -> None:
        """
        Sets the value for force_original_aspect_ratio.

        Args:
            value (int): The value to set.
        """
        ...
    def getForce_original_aspect_ratio(self) -> int:
        """
        Gets the value for force_original_aspect_ratio.

        Returns:
            int: The current value.
        """
        ...
    def setForce_divisible_by(self, value: int) -> None:
        """
        Sets the value for force_divisible_by.

        Args:
            value (int): The value to set.
        """
        ...
    def getForce_divisible_by(self) -> int:
        """
        Gets the value for force_divisible_by.

        Returns:
            int: The current value.
        """
        ...
    def setParam0(self, value: float) -> None:
        """
        Sets the value for param0.

        Args:
            value (float): The value to set.
        """
        ...
    def getParam0(self) -> float:
        """
        Gets the value for param0.

        Returns:
            float: The current value.
        """
        ...
    def setParam1(self, value: float) -> None:
        """
        Sets the value for param1.

        Args:
            value (float): The value to set.
        """
        ...
    def getParam1(self) -> float:
        """
        Gets the value for param1.

        Returns:
            float: The current value.
        """
        ...
    def setEval(self, value: int) -> None:
        """
        Sets the value for eval.

        Args:
            value (int): The value to set.
        """
        ...
    def getEval(self) -> int:
        """
        Gets the value for eval.

        Returns:
            int: The current value.
        """
        ...

class Scdet(FilterBase):
    """
    Detect video scene change
    """
    def setThreshold(self, value: float) -> None:
        """
        Sets the value for threshold.

        Args:
            value (float): The value to set.
        """
        ...
    def getThreshold(self) -> float:
        """
        Gets the value for threshold.

        Returns:
            float: The current value.
        """
        ...
    def setSc_pass(self, value: bool) -> None:
        """
        Sets the value for sc_pass.

        Args:
            value (bool): The value to set.
        """
        ...
    def getSc_pass(self) -> bool:
        """
        Gets the value for sc_pass.

        Returns:
            bool: The current value.
        """
        ...

class Scharr(FilterBase):
    """
    Apply scharr operator.
    """
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setScale(self, value: float) -> None:
        """
        Sets the value for scale.

        Args:
            value (float): The value to set.
        """
        ...
    def getScale(self) -> float:
        """
        Gets the value for scale.

        Returns:
            float: The current value.
        """
        ...
    def setDelta(self, value: float) -> None:
        """
        Sets the value for delta.

        Args:
            value (float): The value to set.
        """
        ...
    def getDelta(self) -> float:
        """
        Gets the value for delta.

        Returns:
            float: The current value.
        """
        ...

class Scroll(FilterBase):
    """
    Scroll input video.
    """
    def setHorizontal(self, value: float) -> None:
        """
        Sets the value for horizontal.

        Args:
            value (float): The value to set.
        """
        ...
    def getHorizontal(self) -> float:
        """
        Gets the value for horizontal.

        Returns:
            float: The current value.
        """
        ...
    def setVertical(self, value: float) -> None:
        """
        Sets the value for vertical.

        Args:
            value (float): The value to set.
        """
        ...
    def getVertical(self) -> float:
        """
        Gets the value for vertical.

        Returns:
            float: The current value.
        """
        ...
    def setHpos(self, value: float) -> None:
        """
        Sets the value for hpos.

        Args:
            value (float): The value to set.
        """
        ...
    def getHpos(self) -> float:
        """
        Gets the value for hpos.

        Returns:
            float: The current value.
        """
        ...
    def setVpos(self, value: float) -> None:
        """
        Sets the value for vpos.

        Args:
            value (float): The value to set.
        """
        ...
    def getVpos(self) -> float:
        """
        Gets the value for vpos.

        Returns:
            float: The current value.
        """
        ...

class Segment(FilterBase):
    """
    Segment video stream.
    """
    def setTimestamps(self, value: Any) -> None:
        """
        Sets the value for timestamps.

        Args:
            value (Any): The value to set.
        """
        ...
    def getTimestamps(self) -> Any:
        """
        Gets the value for timestamps.

        Returns:
            Any: The current value.
        """
        ...
    def setFrames(self, value: Any) -> None:
        """
        Sets the value for frames.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFrames(self) -> Any:
        """
        Gets the value for frames.

        Returns:
            Any: The current value.
        """
        ...

class Select(FilterBase):
    """
    Select video frames to pass in output.
    """
    def setExpr(self, value: Any) -> None:
        """
        Sets the value for expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getExpr(self) -> Any:
        """
        Gets the value for expr.

        Returns:
            Any: The current value.
        """
        ...
    def setOutputs(self, value: int) -> None:
        """
        Sets the value for outputs.

        Args:
            value (int): The value to set.
        """
        ...
    def getOutputs(self) -> int:
        """
        Gets the value for outputs.

        Returns:
            int: The current value.
        """
        ...

class Selectivecolor(FilterBase):
    """
    Apply CMYK adjustments to specific color ranges.
    """
    def setCorrection_method(self, value: int) -> None:
        """
        Sets the value for correction_method.

        Args:
            value (int): The value to set.
        """
        ...
    def getCorrection_method(self) -> int:
        """
        Gets the value for correction_method.

        Returns:
            int: The current value.
        """
        ...
    def setReds(self, value: Any) -> None:
        """
        Sets the value for reds.

        Args:
            value (Any): The value to set.
        """
        ...
    def getReds(self) -> Any:
        """
        Gets the value for reds.

        Returns:
            Any: The current value.
        """
        ...
    def setYellows(self, value: Any) -> None:
        """
        Sets the value for yellows.

        Args:
            value (Any): The value to set.
        """
        ...
    def getYellows(self) -> Any:
        """
        Gets the value for yellows.

        Returns:
            Any: The current value.
        """
        ...
    def setGreens(self, value: Any) -> None:
        """
        Sets the value for greens.

        Args:
            value (Any): The value to set.
        """
        ...
    def getGreens(self) -> Any:
        """
        Gets the value for greens.

        Returns:
            Any: The current value.
        """
        ...
    def setCyans(self, value: Any) -> None:
        """
        Sets the value for cyans.

        Args:
            value (Any): The value to set.
        """
        ...
    def getCyans(self) -> Any:
        """
        Gets the value for cyans.

        Returns:
            Any: The current value.
        """
        ...
    def setBlues(self, value: Any) -> None:
        """
        Sets the value for blues.

        Args:
            value (Any): The value to set.
        """
        ...
    def getBlues(self) -> Any:
        """
        Gets the value for blues.

        Returns:
            Any: The current value.
        """
        ...
    def setMagentas(self, value: Any) -> None:
        """
        Sets the value for magentas.

        Args:
            value (Any): The value to set.
        """
        ...
    def getMagentas(self) -> Any:
        """
        Gets the value for magentas.

        Returns:
            Any: The current value.
        """
        ...
    def setWhites(self, value: Any) -> None:
        """
        Sets the value for whites.

        Args:
            value (Any): The value to set.
        """
        ...
    def getWhites(self) -> Any:
        """
        Gets the value for whites.

        Returns:
            Any: The current value.
        """
        ...
    def setNeutrals(self, value: Any) -> None:
        """
        Sets the value for neutrals.

        Args:
            value (Any): The value to set.
        """
        ...
    def getNeutrals(self) -> Any:
        """
        Gets the value for neutrals.

        Returns:
            Any: The current value.
        """
        ...
    def setBlacks(self, value: Any) -> None:
        """
        Sets the value for blacks.

        Args:
            value (Any): The value to set.
        """
        ...
    def getBlacks(self) -> Any:
        """
        Gets the value for blacks.

        Returns:
            Any: The current value.
        """
        ...
    def setPsfile(self, value: Any) -> None:
        """
        Sets the value for psfile.

        Args:
            value (Any): The value to set.
        """
        ...
    def getPsfile(self) -> Any:
        """
        Gets the value for psfile.

        Returns:
            Any: The current value.
        """
        ...

class Separatefields(FilterBase):
    """
    Split input video frames into fields.
    """
    pass

class Setdar(FilterBase):
    """
    Set the frame display aspect ratio.
    """
    def setRatio(self, value: Any) -> None:
        """
        Sets the value for ratio.

        Args:
            value (Any): The value to set.
        """
        ...
    def getRatio(self) -> Any:
        """
        Gets the value for ratio.

        Returns:
            Any: The current value.
        """
        ...
    def setMax(self, value: int) -> None:
        """
        Sets the value for max.

        Args:
            value (int): The value to set.
        """
        ...
    def getMax(self) -> int:
        """
        Gets the value for max.

        Returns:
            int: The current value.
        """
        ...

class Setfield(FilterBase):
    """
    Force field for the output video frame.
    """
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...

class Setparams(FilterBase):
    """
    Force field, or color property for the output video frame.
    """
    def setField_mode(self, value: int) -> None:
        """
        Sets the value for field_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getField_mode(self) -> int:
        """
        Gets the value for field_mode.

        Returns:
            int: The current value.
        """
        ...
    def setRange(self, value: int) -> None:
        """
        Sets the value for range.

        Args:
            value (int): The value to set.
        """
        ...
    def getRange(self) -> int:
        """
        Gets the value for range.

        Returns:
            int: The current value.
        """
        ...
    def setColor_primaries(self, value: int) -> None:
        """
        Sets the value for color_primaries.

        Args:
            value (int): The value to set.
        """
        ...
    def getColor_primaries(self) -> int:
        """
        Gets the value for color_primaries.

        Returns:
            int: The current value.
        """
        ...
    def setColor_trc(self, value: int) -> None:
        """
        Sets the value for color_trc.

        Args:
            value (int): The value to set.
        """
        ...
    def getColor_trc(self) -> int:
        """
        Gets the value for color_trc.

        Returns:
            int: The current value.
        """
        ...
    def setColorspace(self, value: int) -> None:
        """
        Sets the value for colorspace.

        Args:
            value (int): The value to set.
        """
        ...
    def getColorspace(self) -> int:
        """
        Gets the value for colorspace.

        Returns:
            int: The current value.
        """
        ...

class Setpts(FilterBase):
    """
    Set PTS for the output video frame.
    """
    def setExpr(self, value: Any) -> None:
        """
        Sets the value for expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getExpr(self) -> Any:
        """
        Gets the value for expr.

        Returns:
            Any: The current value.
        """
        ...

class Setrange(FilterBase):
    """
    Force color range for the output video frame.
    """
    def setRange(self, value: int) -> None:
        """
        Sets the value for range.

        Args:
            value (int): The value to set.
        """
        ...
    def getRange(self) -> int:
        """
        Gets the value for range.

        Returns:
            int: The current value.
        """
        ...

class Setsar(FilterBase):
    """
    Set the pixel sample aspect ratio.
    """
    def setRatio(self, value: Any) -> None:
        """
        Sets the value for ratio.

        Args:
            value (Any): The value to set.
        """
        ...
    def getRatio(self) -> Any:
        """
        Gets the value for ratio.

        Returns:
            Any: The current value.
        """
        ...
    def setMax(self, value: int) -> None:
        """
        Sets the value for max.

        Args:
            value (int): The value to set.
        """
        ...
    def getMax(self) -> int:
        """
        Gets the value for max.

        Returns:
            int: The current value.
        """
        ...

class Settb(FilterBase):
    """
    Set timebase for the video output link.
    """
    def setExpr(self, value: Any) -> None:
        """
        Sets the value for expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getExpr(self) -> Any:
        """
        Gets the value for expr.

        Returns:
            Any: The current value.
        """
        ...

class Shear(FilterBase):
    """
    Shear transform the input image.
    """
    def setShx(self, value: float) -> None:
        """
        Sets the value for shx.

        Args:
            value (float): The value to set.
        """
        ...
    def getShx(self) -> float:
        """
        Gets the value for shx.

        Returns:
            float: The current value.
        """
        ...
    def setShy(self, value: float) -> None:
        """
        Sets the value for shy.

        Args:
            value (float): The value to set.
        """
        ...
    def getShy(self) -> float:
        """
        Gets the value for shy.

        Returns:
            float: The current value.
        """
        ...
    def setFillcolor(self, value: Any) -> None:
        """
        Sets the value for fillcolor.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFillcolor(self) -> Any:
        """
        Gets the value for fillcolor.

        Returns:
            Any: The current value.
        """
        ...
    def setInterp(self, value: int) -> None:
        """
        Sets the value for interp.

        Args:
            value (int): The value to set.
        """
        ...
    def getInterp(self) -> int:
        """
        Gets the value for interp.

        Returns:
            int: The current value.
        """
        ...

class Showinfo(FilterBase):
    """
    Show textual information for each video frame.
    """
    def setChecksum(self, value: bool) -> None:
        """
        Sets the value for checksum.

        Args:
            value (bool): The value to set.
        """
        ...
    def getChecksum(self) -> bool:
        """
        Gets the value for checksum.

        Returns:
            bool: The current value.
        """
        ...
    def setUdu_sei_as_ascii(self, value: bool) -> None:
        """
        Sets the value for udu_sei_as_ascii.

        Args:
            value (bool): The value to set.
        """
        ...
    def getUdu_sei_as_ascii(self) -> bool:
        """
        Gets the value for udu_sei_as_ascii.

        Returns:
            bool: The current value.
        """
        ...

class Showpalette(FilterBase):
    """
    Display frame palette.
    """
    def setPixelBoxSize(self, value: int) -> None:
        """
        Sets the value for pixelboxsize.

        Args:
            value (int): The value to set.
        """
        ...
    def getPixelBoxSize(self) -> int:
        """
        Gets the value for pixelboxsize.

        Returns:
            int: The current value.
        """
        ...

class Shuffleframes(FilterBase):
    """
    Shuffle video frames.
    """
    def setMapping(self, value: Any) -> None:
        """
        Sets the value for mapping.

        Args:
            value (Any): The value to set.
        """
        ...
    def getMapping(self) -> Any:
        """
        Gets the value for mapping.

        Returns:
            Any: The current value.
        """
        ...

class Shufflepixels(FilterBase):
    """
    Shuffle video pixels.
    """
    def setDirection(self, value: int) -> None:
        """
        Sets the value for direction.

        Args:
            value (int): The value to set.
        """
        ...
    def getDirection(self) -> int:
        """
        Gets the value for direction.

        Returns:
            int: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setWidth(self, value: int) -> None:
        """
        Sets the value for width.

        Args:
            value (int): The value to set.
        """
        ...
    def getWidth(self) -> int:
        """
        Gets the value for width.

        Returns:
            int: The current value.
        """
        ...
    def setHeight(self, value: int) -> None:
        """
        Sets the value for height.

        Args:
            value (int): The value to set.
        """
        ...
    def getHeight(self) -> int:
        """
        Gets the value for height.

        Returns:
            int: The current value.
        """
        ...
    def setSeed(self, value: int) -> None:
        """
        Sets the value for seed.

        Args:
            value (int): The value to set.
        """
        ...
    def getSeed(self) -> int:
        """
        Gets the value for seed.

        Returns:
            int: The current value.
        """
        ...

class Shuffleplanes(FilterBase):
    """
    Shuffle video planes.
    """
    def setMap0(self, value: int) -> None:
        """
        Sets the value for map0.

        Args:
            value (int): The value to set.
        """
        ...
    def getMap0(self) -> int:
        """
        Gets the value for map0.

        Returns:
            int: The current value.
        """
        ...
    def setMap1(self, value: int) -> None:
        """
        Sets the value for map1.

        Args:
            value (int): The value to set.
        """
        ...
    def getMap1(self) -> int:
        """
        Gets the value for map1.

        Returns:
            int: The current value.
        """
        ...
    def setMap2(self, value: int) -> None:
        """
        Sets the value for map2.

        Args:
            value (int): The value to set.
        """
        ...
    def getMap2(self) -> int:
        """
        Gets the value for map2.

        Returns:
            int: The current value.
        """
        ...
    def setMap3(self, value: int) -> None:
        """
        Sets the value for map3.

        Args:
            value (int): The value to set.
        """
        ...
    def getMap3(self) -> int:
        """
        Gets the value for map3.

        Returns:
            int: The current value.
        """
        ...

class Sidedata(FilterBase):
    """
    Manipulate video frame side data.
    """
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setType(self, value: int) -> None:
        """
        Sets the value for type.

        Args:
            value (int): The value to set.
        """
        ...
    def getType(self) -> int:
        """
        Gets the value for type.

        Returns:
            int: The current value.
        """
        ...

class Signalstats(FilterBase):
    """
    Generate statistics from video analysis.
    """
    def setStat(self, value: int) -> None:
        """
        Sets the value for stat.

        Args:
            value (int): The value to set.
        """
        ...
    def getStat(self) -> int:
        """
        Gets the value for stat.

        Returns:
            int: The current value.
        """
        ...
    def setOut(self, value: int) -> None:
        """
        Sets the value for out.

        Args:
            value (int): The value to set.
        """
        ...
    def getOut(self) -> int:
        """
        Gets the value for out.

        Returns:
            int: The current value.
        """
        ...
    def setColor(self, value: Any) -> None:
        """
        Sets the value for color.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColor(self) -> Any:
        """
        Gets the value for color.

        Returns:
            Any: The current value.
        """
        ...

class Siti(FilterBase):
    """
    Calculate spatial information (SI) and temporal information (TI).
    """
    def setPrint_summary(self, value: bool) -> None:
        """
        Sets the value for print_summary.

        Args:
            value (bool): The value to set.
        """
        ...
    def getPrint_summary(self) -> bool:
        """
        Gets the value for print_summary.

        Returns:
            bool: The current value.
        """
        ...

class Sobel(FilterBase):
    """
    Apply sobel operator.
    """
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setScale(self, value: float) -> None:
        """
        Sets the value for scale.

        Args:
            value (float): The value to set.
        """
        ...
    def getScale(self) -> float:
        """
        Gets the value for scale.

        Returns:
            float: The current value.
        """
        ...
    def setDelta(self, value: float) -> None:
        """
        Sets the value for delta.

        Args:
            value (float): The value to set.
        """
        ...
    def getDelta(self) -> float:
        """
        Gets the value for delta.

        Returns:
            float: The current value.
        """
        ...

class Sr(FilterBase):
    """
    Apply DNN-based image super resolution to the input.
    """
    def setDnn_backend(self, value: int) -> None:
        """
        Sets the value for dnn_backend.

        Args:
            value (int): The value to set.
        """
        ...
    def getDnn_backend(self) -> int:
        """
        Gets the value for dnn_backend.

        Returns:
            int: The current value.
        """
        ...
    def setScale_factor(self, value: int) -> None:
        """
        Sets the value for scale_factor.

        Args:
            value (int): The value to set.
        """
        ...
    def getScale_factor(self) -> int:
        """
        Gets the value for scale_factor.

        Returns:
            int: The current value.
        """
        ...
    def setModel(self, value: Any) -> None:
        """
        Sets the value for model.

        Args:
            value (Any): The value to set.
        """
        ...
    def getModel(self) -> Any:
        """
        Gets the value for model.

        Returns:
            Any: The current value.
        """
        ...
    def setInput(self, value: Any) -> None:
        """
        Sets the value for input.

        Args:
            value (Any): The value to set.
        """
        ...
    def getInput(self) -> Any:
        """
        Gets the value for input.

        Returns:
            Any: The current value.
        """
        ...
    def setOutput(self, value: Any) -> None:
        """
        Sets the value for output.

        Args:
            value (Any): The value to set.
        """
        ...
    def getOutput(self) -> Any:
        """
        Gets the value for output.

        Returns:
            Any: The current value.
        """
        ...

class Ssim(FilterBase):
    """
    Calculate the SSIM between two video streams.
    """
    def setStats_file(self, value: Any) -> None:
        """
        Sets the value for stats_file.

        Args:
            value (Any): The value to set.
        """
        ...
    def getStats_file(self) -> Any:
        """
        Gets the value for stats_file.

        Returns:
            Any: The current value.
        """
        ...

class Ssim360(FilterBase):
    """
    Calculate the SSIM between two 360 video streams.
    """
    def setStats_file(self, value: Any) -> None:
        """
        Sets the value for stats_file.

        Args:
            value (Any): The value to set.
        """
        ...
    def getStats_file(self) -> Any:
        """
        Gets the value for stats_file.

        Returns:
            Any: The current value.
        """
        ...
    def setCompute_chroma(self, value: int) -> None:
        """
        Sets the value for compute_chroma.

        Args:
            value (int): The value to set.
        """
        ...
    def getCompute_chroma(self) -> int:
        """
        Gets the value for compute_chroma.

        Returns:
            int: The current value.
        """
        ...
    def setFrame_skip_ratio(self, value: int) -> None:
        """
        Sets the value for frame_skip_ratio.

        Args:
            value (int): The value to set.
        """
        ...
    def getFrame_skip_ratio(self) -> int:
        """
        Gets the value for frame_skip_ratio.

        Returns:
            int: The current value.
        """
        ...
    def setRef_projection(self, value: int) -> None:
        """
        Sets the value for ref_projection.

        Args:
            value (int): The value to set.
        """
        ...
    def getRef_projection(self) -> int:
        """
        Gets the value for ref_projection.

        Returns:
            int: The current value.
        """
        ...
    def setMain_projection(self, value: int) -> None:
        """
        Sets the value for main_projection.

        Args:
            value (int): The value to set.
        """
        ...
    def getMain_projection(self) -> int:
        """
        Gets the value for main_projection.

        Returns:
            int: The current value.
        """
        ...
    def setRef_stereo(self, value: int) -> None:
        """
        Sets the value for ref_stereo.

        Args:
            value (int): The value to set.
        """
        ...
    def getRef_stereo(self) -> int:
        """
        Gets the value for ref_stereo.

        Returns:
            int: The current value.
        """
        ...
    def setMain_stereo(self, value: int) -> None:
        """
        Sets the value for main_stereo.

        Args:
            value (int): The value to set.
        """
        ...
    def getMain_stereo(self) -> int:
        """
        Gets the value for main_stereo.

        Returns:
            int: The current value.
        """
        ...
    def setRef_pad(self, value: float) -> None:
        """
        Sets the value for ref_pad.

        Args:
            value (float): The value to set.
        """
        ...
    def getRef_pad(self) -> float:
        """
        Gets the value for ref_pad.

        Returns:
            float: The current value.
        """
        ...
    def setMain_pad(self, value: float) -> None:
        """
        Sets the value for main_pad.

        Args:
            value (float): The value to set.
        """
        ...
    def getMain_pad(self) -> float:
        """
        Gets the value for main_pad.

        Returns:
            float: The current value.
        """
        ...
    def setUse_tape(self, value: int) -> None:
        """
        Sets the value for use_tape.

        Args:
            value (int): The value to set.
        """
        ...
    def getUse_tape(self) -> int:
        """
        Gets the value for use_tape.

        Returns:
            int: The current value.
        """
        ...
    def setHeatmap_str(self, value: Any) -> None:
        """
        Sets the value for heatmap_str.

        Args:
            value (Any): The value to set.
        """
        ...
    def getHeatmap_str(self) -> Any:
        """
        Gets the value for heatmap_str.

        Returns:
            Any: The current value.
        """
        ...
    def setDefault_heatmap_width(self, value: int) -> None:
        """
        Sets the value for default_heatmap_width.

        Args:
            value (int): The value to set.
        """
        ...
    def getDefault_heatmap_width(self) -> int:
        """
        Gets the value for default_heatmap_width.

        Returns:
            int: The current value.
        """
        ...
    def setDefault_heatmap_height(self, value: int) -> None:
        """
        Sets the value for default_heatmap_height.

        Args:
            value (int): The value to set.
        """
        ...
    def getDefault_heatmap_height(self) -> int:
        """
        Gets the value for default_heatmap_height.

        Returns:
            int: The current value.
        """
        ...

class Swaprect(FilterBase):
    """
    Swap 2 rectangular objects in video.
    """
    def setRectWidth(self, value: Any) -> None:
        """
        Sets the value for rectwidth.

        Args:
            value (Any): The value to set.
        """
        ...
    def getRectWidth(self) -> Any:
        """
        Gets the value for rectwidth.

        Returns:
            Any: The current value.
        """
        ...
    def setRectHeight(self, value: Any) -> None:
        """
        Sets the value for rectheight.

        Args:
            value (Any): The value to set.
        """
        ...
    def getRectHeight(self) -> Any:
        """
        Gets the value for rectheight.

        Returns:
            Any: The current value.
        """
        ...
    def setX1(self, value: Any) -> None:
        """
        Sets the value for x1.

        Args:
            value (Any): The value to set.
        """
        ...
    def getX1(self) -> Any:
        """
        Gets the value for x1.

        Returns:
            Any: The current value.
        """
        ...
    def setY1(self, value: Any) -> None:
        """
        Sets the value for y1.

        Args:
            value (Any): The value to set.
        """
        ...
    def getY1(self) -> Any:
        """
        Gets the value for y1.

        Returns:
            Any: The current value.
        """
        ...
    def setX2(self, value: Any) -> None:
        """
        Sets the value for x2.

        Args:
            value (Any): The value to set.
        """
        ...
    def getX2(self) -> Any:
        """
        Gets the value for x2.

        Returns:
            Any: The current value.
        """
        ...
    def setY2(self, value: Any) -> None:
        """
        Sets the value for y2.

        Args:
            value (Any): The value to set.
        """
        ...
    def getY2(self) -> Any:
        """
        Gets the value for y2.

        Returns:
            Any: The current value.
        """
        ...

class Swapuv(FilterBase):
    """
    Swap U and V components.
    """
    pass

class Tblend(FilterBase):
    """
    Blend successive frames.
    """
    def setC0_mode(self, value: int) -> None:
        """
        Sets the value for c0_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getC0_mode(self) -> int:
        """
        Gets the value for c0_mode.

        Returns:
            int: The current value.
        """
        ...
    def setC1_mode(self, value: int) -> None:
        """
        Sets the value for c1_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getC1_mode(self) -> int:
        """
        Gets the value for c1_mode.

        Returns:
            int: The current value.
        """
        ...
    def setC2_mode(self, value: int) -> None:
        """
        Sets the value for c2_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getC2_mode(self) -> int:
        """
        Gets the value for c2_mode.

        Returns:
            int: The current value.
        """
        ...
    def setC3_mode(self, value: int) -> None:
        """
        Sets the value for c3_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getC3_mode(self) -> int:
        """
        Gets the value for c3_mode.

        Returns:
            int: The current value.
        """
        ...
    def setAll_mode(self, value: int) -> None:
        """
        Sets the value for all_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getAll_mode(self) -> int:
        """
        Gets the value for all_mode.

        Returns:
            int: The current value.
        """
        ...
    def setC0_expr(self, value: Any) -> None:
        """
        Sets the value for c0_expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC0_expr(self) -> Any:
        """
        Gets the value for c0_expr.

        Returns:
            Any: The current value.
        """
        ...
    def setC1_expr(self, value: Any) -> None:
        """
        Sets the value for c1_expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC1_expr(self) -> Any:
        """
        Gets the value for c1_expr.

        Returns:
            Any: The current value.
        """
        ...
    def setC2_expr(self, value: Any) -> None:
        """
        Sets the value for c2_expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC2_expr(self) -> Any:
        """
        Gets the value for c2_expr.

        Returns:
            Any: The current value.
        """
        ...
    def setC3_expr(self, value: Any) -> None:
        """
        Sets the value for c3_expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC3_expr(self) -> Any:
        """
        Gets the value for c3_expr.

        Returns:
            Any: The current value.
        """
        ...
    def setAll_expr(self, value: Any) -> None:
        """
        Sets the value for all_expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getAll_expr(self) -> Any:
        """
        Gets the value for all_expr.

        Returns:
            Any: The current value.
        """
        ...
    def setC0_opacity(self, value: float) -> None:
        """
        Sets the value for c0_opacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getC0_opacity(self) -> float:
        """
        Gets the value for c0_opacity.

        Returns:
            float: The current value.
        """
        ...
    def setC1_opacity(self, value: float) -> None:
        """
        Sets the value for c1_opacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getC1_opacity(self) -> float:
        """
        Gets the value for c1_opacity.

        Returns:
            float: The current value.
        """
        ...
    def setC2_opacity(self, value: float) -> None:
        """
        Sets the value for c2_opacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getC2_opacity(self) -> float:
        """
        Gets the value for c2_opacity.

        Returns:
            float: The current value.
        """
        ...
    def setC3_opacity(self, value: float) -> None:
        """
        Sets the value for c3_opacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getC3_opacity(self) -> float:
        """
        Gets the value for c3_opacity.

        Returns:
            float: The current value.
        """
        ...
    def setAll_opacity(self, value: float) -> None:
        """
        Sets the value for all_opacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getAll_opacity(self) -> float:
        """
        Gets the value for all_opacity.

        Returns:
            float: The current value.
        """
        ...

class Telecine(FilterBase):
    """
    Apply a telecine pattern.
    """
    def setFirst_field(self, value: int) -> None:
        """
        Sets the value for first_field.

        Args:
            value (int): The value to set.
        """
        ...
    def getFirst_field(self) -> int:
        """
        Gets the value for first_field.

        Returns:
            int: The current value.
        """
        ...
    def setPattern(self, value: Any) -> None:
        """
        Sets the value for pattern.

        Args:
            value (Any): The value to set.
        """
        ...
    def getPattern(self) -> Any:
        """
        Gets the value for pattern.

        Returns:
            Any: The current value.
        """
        ...

class Thistogram(FilterBase):
    """
    Compute and draw a temporal histogram.
    """
    def setWidth(self, value: int) -> None:
        """
        Sets the value for width.

        Args:
            value (int): The value to set.
        """
        ...
    def getWidth(self) -> int:
        """
        Gets the value for width.

        Returns:
            int: The current value.
        """
        ...
    def setDisplay_mode(self, value: int) -> None:
        """
        Sets the value for display_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getDisplay_mode(self) -> int:
        """
        Gets the value for display_mode.

        Returns:
            int: The current value.
        """
        ...
    def setLevels_mode(self, value: int) -> None:
        """
        Sets the value for levels_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getLevels_mode(self) -> int:
        """
        Gets the value for levels_mode.

        Returns:
            int: The current value.
        """
        ...
    def setComponents(self, value: int) -> None:
        """
        Sets the value for components.

        Args:
            value (int): The value to set.
        """
        ...
    def getComponents(self) -> int:
        """
        Gets the value for components.

        Returns:
            int: The current value.
        """
        ...
    def setBgopacity(self, value: float) -> None:
        """
        Sets the value for bgopacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getBgopacity(self) -> float:
        """
        Gets the value for bgopacity.

        Returns:
            float: The current value.
        """
        ...
    def setEnvelope(self, value: bool) -> None:
        """
        Sets the value for envelope.

        Args:
            value (bool): The value to set.
        """
        ...
    def getEnvelope(self) -> bool:
        """
        Gets the value for envelope.

        Returns:
            bool: The current value.
        """
        ...
    def setEcolor(self, value: Any) -> None:
        """
        Sets the value for ecolor.

        Args:
            value (Any): The value to set.
        """
        ...
    def getEcolor(self) -> Any:
        """
        Gets the value for ecolor.

        Returns:
            Any: The current value.
        """
        ...
    def setSlide(self, value: int) -> None:
        """
        Sets the value for slide.

        Args:
            value (int): The value to set.
        """
        ...
    def getSlide(self) -> int:
        """
        Gets the value for slide.

        Returns:
            int: The current value.
        """
        ...

class Threshold(FilterBase):
    """
    Threshold first video stream using other video streams.
    """
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Thumbnail(FilterBase):
    """
    Select the most representative frame in a given sequence of consecutive frames.
    """
    def setFramesBatchSize(self, value: int) -> None:
        """
        Sets the value for framesbatchsize.

        Args:
            value (int): The value to set.
        """
        ...
    def getFramesBatchSize(self) -> int:
        """
        Gets the value for framesbatchsize.

        Returns:
            int: The current value.
        """
        ...
    def setLog(self, value: int) -> None:
        """
        Sets the value for log.

        Args:
            value (int): The value to set.
        """
        ...
    def getLog(self) -> int:
        """
        Gets the value for log.

        Returns:
            int: The current value.
        """
        ...

class Tile(FilterBase):
    """
    Tile several successive frames together.
    """
    def setLayout(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for layout.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getLayout(self) -> Tuple[int, int]:
        """
        Gets the value for layout.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setNb_frames(self, value: int) -> None:
        """
        Sets the value for nb_frames.

        Args:
            value (int): The value to set.
        """
        ...
    def getNb_frames(self) -> int:
        """
        Gets the value for nb_frames.

        Returns:
            int: The current value.
        """
        ...
    def setMargin(self, value: int) -> None:
        """
        Sets the value for margin.

        Args:
            value (int): The value to set.
        """
        ...
    def getMargin(self) -> int:
        """
        Gets the value for margin.

        Returns:
            int: The current value.
        """
        ...
    def setPadding(self, value: int) -> None:
        """
        Sets the value for padding.

        Args:
            value (int): The value to set.
        """
        ...
    def getPadding(self) -> int:
        """
        Gets the value for padding.

        Returns:
            int: The current value.
        """
        ...
    def setColor(self, value: Any) -> None:
        """
        Sets the value for color.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColor(self) -> Any:
        """
        Gets the value for color.

        Returns:
            Any: The current value.
        """
        ...
    def setOverlap(self, value: int) -> None:
        """
        Sets the value for overlap.

        Args:
            value (int): The value to set.
        """
        ...
    def getOverlap(self) -> int:
        """
        Gets the value for overlap.

        Returns:
            int: The current value.
        """
        ...
    def setInit_padding(self, value: int) -> None:
        """
        Sets the value for init_padding.

        Args:
            value (int): The value to set.
        """
        ...
    def getInit_padding(self) -> int:
        """
        Gets the value for init_padding.

        Returns:
            int: The current value.
        """
        ...

class Tiltandshift(FilterBase):
    """
    Generate a tilt-and-shift'd video.
    """
    def setTilt(self, value: int) -> None:
        """
        Sets the value for tilt.

        Args:
            value (int): The value to set.
        """
        ...
    def getTilt(self) -> int:
        """
        Gets the value for tilt.

        Returns:
            int: The current value.
        """
        ...
    def setStart(self, value: int) -> None:
        """
        Sets the value for start.

        Args:
            value (int): The value to set.
        """
        ...
    def getStart(self) -> int:
        """
        Gets the value for start.

        Returns:
            int: The current value.
        """
        ...
    def setEnd(self, value: int) -> None:
        """
        Sets the value for end.

        Args:
            value (int): The value to set.
        """
        ...
    def getEnd(self) -> int:
        """
        Gets the value for end.

        Returns:
            int: The current value.
        """
        ...
    def setHold(self, value: int) -> None:
        """
        Sets the value for hold.

        Args:
            value (int): The value to set.
        """
        ...
    def getHold(self) -> int:
        """
        Gets the value for hold.

        Returns:
            int: The current value.
        """
        ...
    def setPad(self, value: int) -> None:
        """
        Sets the value for pad.

        Args:
            value (int): The value to set.
        """
        ...
    def getPad(self) -> int:
        """
        Gets the value for pad.

        Returns:
            int: The current value.
        """
        ...

class Tlut2(FilterBase):
    """
    Compute and apply a lookup table from two successive frames.
    """
    def setC0(self, value: Any) -> None:
        """
        Sets the value for c0.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC0(self) -> Any:
        """
        Gets the value for c0.

        Returns:
            Any: The current value.
        """
        ...
    def setC1(self, value: Any) -> None:
        """
        Sets the value for c1.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC1(self) -> Any:
        """
        Gets the value for c1.

        Returns:
            Any: The current value.
        """
        ...
    def setC2(self, value: Any) -> None:
        """
        Sets the value for c2.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC2(self) -> Any:
        """
        Gets the value for c2.

        Returns:
            Any: The current value.
        """
        ...
    def setC3(self, value: Any) -> None:
        """
        Sets the value for c3.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC3(self) -> Any:
        """
        Gets the value for c3.

        Returns:
            Any: The current value.
        """
        ...

class Tmedian(FilterBase):
    """
    Pick median pixels from successive frames.
    """
    def setRadius(self, value: int) -> None:
        """
        Sets the value for radius.

        Args:
            value (int): The value to set.
        """
        ...
    def getRadius(self) -> int:
        """
        Gets the value for radius.

        Returns:
            int: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setPercentile(self, value: float) -> None:
        """
        Sets the value for percentile.

        Args:
            value (float): The value to set.
        """
        ...
    def getPercentile(self) -> float:
        """
        Gets the value for percentile.

        Returns:
            float: The current value.
        """
        ...

class Tmidequalizer(FilterBase):
    """
    Apply Temporal Midway Equalization.
    """
    def setRadius(self, value: int) -> None:
        """
        Sets the value for radius.

        Args:
            value (int): The value to set.
        """
        ...
    def getRadius(self) -> int:
        """
        Gets the value for radius.

        Returns:
            int: The current value.
        """
        ...
    def setSigma(self, value: float) -> None:
        """
        Sets the value for sigma.

        Args:
            value (float): The value to set.
        """
        ...
    def getSigma(self) -> float:
        """
        Gets the value for sigma.

        Returns:
            float: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Tmix(FilterBase):
    """
    Mix successive video frames.
    """
    def setFrames(self, value: int) -> None:
        """
        Sets the value for frames.

        Args:
            value (int): The value to set.
        """
        ...
    def getFrames(self) -> int:
        """
        Gets the value for frames.

        Returns:
            int: The current value.
        """
        ...
    def setWeights(self, value: Any) -> None:
        """
        Sets the value for weights.

        Args:
            value (Any): The value to set.
        """
        ...
    def getWeights(self) -> Any:
        """
        Gets the value for weights.

        Returns:
            Any: The current value.
        """
        ...
    def setScale(self, value: float) -> None:
        """
        Sets the value for scale.

        Args:
            value (float): The value to set.
        """
        ...
    def getScale(self) -> float:
        """
        Gets the value for scale.

        Returns:
            float: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Tonemap(FilterBase):
    """
    Conversion to/from different dynamic ranges.
    """
    def setTonemap(self, value: int) -> None:
        """
        Sets the value for tonemap.

        Args:
            value (int): The value to set.
        """
        ...
    def getTonemap(self) -> int:
        """
        Gets the value for tonemap.

        Returns:
            int: The current value.
        """
        ...
    def setParam(self, value: float) -> None:
        """
        Sets the value for param.

        Args:
            value (float): The value to set.
        """
        ...
    def getParam(self) -> float:
        """
        Gets the value for param.

        Returns:
            float: The current value.
        """
        ...
    def setDesat(self, value: float) -> None:
        """
        Sets the value for desat.

        Args:
            value (float): The value to set.
        """
        ...
    def getDesat(self) -> float:
        """
        Gets the value for desat.

        Returns:
            float: The current value.
        """
        ...
    def setPeak(self, value: float) -> None:
        """
        Sets the value for peak.

        Args:
            value (float): The value to set.
        """
        ...
    def getPeak(self) -> float:
        """
        Gets the value for peak.

        Returns:
            float: The current value.
        """
        ...

class Tpad(FilterBase):
    """
    Temporarily pad video frames.
    """
    def setStart(self, value: int) -> None:
        """
        Sets the value for start.

        Args:
            value (int): The value to set.
        """
        ...
    def getStart(self) -> int:
        """
        Gets the value for start.

        Returns:
            int: The current value.
        """
        ...
    def setStop(self, value: int) -> None:
        """
        Sets the value for stop.

        Args:
            value (int): The value to set.
        """
        ...
    def getStop(self) -> int:
        """
        Gets the value for stop.

        Returns:
            int: The current value.
        """
        ...
    def setStart_mode(self, value: int) -> None:
        """
        Sets the value for start_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getStart_mode(self) -> int:
        """
        Gets the value for start_mode.

        Returns:
            int: The current value.
        """
        ...
    def setStop_mode(self, value: int) -> None:
        """
        Sets the value for stop_mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getStop_mode(self) -> int:
        """
        Gets the value for stop_mode.

        Returns:
            int: The current value.
        """
        ...
    def setStart_duration(self, value: int) -> None:
        """
        Sets the value for start_duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getStart_duration(self) -> int:
        """
        Gets the value for start_duration.

        Returns:
            int: The current value.
        """
        ...
    def setStop_duration(self, value: int) -> None:
        """
        Sets the value for stop_duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getStop_duration(self) -> int:
        """
        Gets the value for stop_duration.

        Returns:
            int: The current value.
        """
        ...
    def setColor(self, value: Any) -> None:
        """
        Sets the value for color.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColor(self) -> Any:
        """
        Gets the value for color.

        Returns:
            Any: The current value.
        """
        ...

class Transpose(FilterBase):
    """
    Transpose input video.
    """
    def setDir(self, value: int) -> None:
        """
        Sets the value for dir.

        Args:
            value (int): The value to set.
        """
        ...
    def getDir(self) -> int:
        """
        Gets the value for dir.

        Returns:
            int: The current value.
        """
        ...
    def setPassthrough(self, value: int) -> None:
        """
        Sets the value for passthrough.

        Args:
            value (int): The value to set.
        """
        ...
    def getPassthrough(self) -> int:
        """
        Gets the value for passthrough.

        Returns:
            int: The current value.
        """
        ...

class Trim(FilterBase):
    """
    Pick one continuous section from the input, drop the rest.
    """
    def setStarti(self, value: int) -> None:
        """
        Sets the value for starti.

        Args:
            value (int): The value to set.
        """
        ...
    def getStarti(self) -> int:
        """
        Gets the value for starti.

        Returns:
            int: The current value.
        """
        ...
    def setEndi(self, value: int) -> None:
        """
        Sets the value for endi.

        Args:
            value (int): The value to set.
        """
        ...
    def getEndi(self) -> int:
        """
        Gets the value for endi.

        Returns:
            int: The current value.
        """
        ...
    def setStart_pts(self, value: int) -> None:
        """
        Sets the value for start_pts.

        Args:
            value (int): The value to set.
        """
        ...
    def getStart_pts(self) -> int:
        """
        Gets the value for start_pts.

        Returns:
            int: The current value.
        """
        ...
    def setEnd_pts(self, value: int) -> None:
        """
        Sets the value for end_pts.

        Args:
            value (int): The value to set.
        """
        ...
    def getEnd_pts(self) -> int:
        """
        Gets the value for end_pts.

        Returns:
            int: The current value.
        """
        ...
    def setDurationi(self, value: int) -> None:
        """
        Sets the value for durationi.

        Args:
            value (int): The value to set.
        """
        ...
    def getDurationi(self) -> int:
        """
        Gets the value for durationi.

        Returns:
            int: The current value.
        """
        ...
    def setStart_frame(self, value: int) -> None:
        """
        Sets the value for start_frame.

        Args:
            value (int): The value to set.
        """
        ...
    def getStart_frame(self) -> int:
        """
        Gets the value for start_frame.

        Returns:
            int: The current value.
        """
        ...
    def setEnd_frame(self, value: int) -> None:
        """
        Sets the value for end_frame.

        Args:
            value (int): The value to set.
        """
        ...
    def getEnd_frame(self) -> int:
        """
        Gets the value for end_frame.

        Returns:
            int: The current value.
        """
        ...

class Unpremultiply(FilterBase):
    """
    UnPreMultiply first stream with first plane of second stream.
    """
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setInplace(self, value: bool) -> None:
        """
        Sets the value for inplace.

        Args:
            value (bool): The value to set.
        """
        ...
    def getInplace(self) -> bool:
        """
        Gets the value for inplace.

        Returns:
            bool: The current value.
        """
        ...

class Unsharp(FilterBase):
    """
    Sharpen or blur the input video.
    """
    def setLuma_msize_x(self, value: int) -> None:
        """
        Sets the value for luma_msize_x.

        Args:
            value (int): The value to set.
        """
        ...
    def getLuma_msize_x(self) -> int:
        """
        Gets the value for luma_msize_x.

        Returns:
            int: The current value.
        """
        ...
    def setLuma_msize_y(self, value: int) -> None:
        """
        Sets the value for luma_msize_y.

        Args:
            value (int): The value to set.
        """
        ...
    def getLuma_msize_y(self) -> int:
        """
        Gets the value for luma_msize_y.

        Returns:
            int: The current value.
        """
        ...
    def setLuma_amount(self, value: float) -> None:
        """
        Sets the value for luma_amount.

        Args:
            value (float): The value to set.
        """
        ...
    def getLuma_amount(self) -> float:
        """
        Gets the value for luma_amount.

        Returns:
            float: The current value.
        """
        ...
    def setChroma_msize_x(self, value: int) -> None:
        """
        Sets the value for chroma_msize_x.

        Args:
            value (int): The value to set.
        """
        ...
    def getChroma_msize_x(self) -> int:
        """
        Gets the value for chroma_msize_x.

        Returns:
            int: The current value.
        """
        ...
    def setChroma_msize_y(self, value: int) -> None:
        """
        Sets the value for chroma_msize_y.

        Args:
            value (int): The value to set.
        """
        ...
    def getChroma_msize_y(self) -> int:
        """
        Gets the value for chroma_msize_y.

        Returns:
            int: The current value.
        """
        ...
    def setChroma_amount(self, value: float) -> None:
        """
        Sets the value for chroma_amount.

        Args:
            value (float): The value to set.
        """
        ...
    def getChroma_amount(self) -> float:
        """
        Gets the value for chroma_amount.

        Returns:
            float: The current value.
        """
        ...
    def setAlpha_msize_x(self, value: int) -> None:
        """
        Sets the value for alpha_msize_x.

        Args:
            value (int): The value to set.
        """
        ...
    def getAlpha_msize_x(self) -> int:
        """
        Gets the value for alpha_msize_x.

        Returns:
            int: The current value.
        """
        ...
    def setAlpha_msize_y(self, value: int) -> None:
        """
        Sets the value for alpha_msize_y.

        Args:
            value (int): The value to set.
        """
        ...
    def getAlpha_msize_y(self) -> int:
        """
        Gets the value for alpha_msize_y.

        Returns:
            int: The current value.
        """
        ...
    def setAlpha_amount(self, value: float) -> None:
        """
        Sets the value for alpha_amount.

        Args:
            value (float): The value to set.
        """
        ...
    def getAlpha_amount(self) -> float:
        """
        Gets the value for alpha_amount.

        Returns:
            float: The current value.
        """
        ...

class Untile(FilterBase):
    """
    Untile a frame into a sequence of frames.
    """
    def setLayout(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for layout.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getLayout(self) -> Tuple[int, int]:
        """
        Gets the value for layout.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...

class V360(FilterBase):
    """
    Convert 360 projection of video.
    """
    def setInput(self, value: int) -> None:
        """
        Sets the value for input.

        Args:
            value (int): The value to set.
        """
        ...
    def getInput(self) -> int:
        """
        Gets the value for input.

        Returns:
            int: The current value.
        """
        ...
    def setOutput(self, value: int) -> None:
        """
        Sets the value for output.

        Args:
            value (int): The value to set.
        """
        ...
    def getOutput(self) -> int:
        """
        Gets the value for output.

        Returns:
            int: The current value.
        """
        ...
    def setInterp(self, value: int) -> None:
        """
        Sets the value for interp.

        Args:
            value (int): The value to set.
        """
        ...
    def getInterp(self) -> int:
        """
        Gets the value for interp.

        Returns:
            int: The current value.
        """
        ...
    def setOutputWidth(self, value: int) -> None:
        """
        Sets the value for outputwidth.

        Args:
            value (int): The value to set.
        """
        ...
    def getOutputWidth(self) -> int:
        """
        Gets the value for outputwidth.

        Returns:
            int: The current value.
        """
        ...
    def setOutputHeight(self, value: int) -> None:
        """
        Sets the value for outputheight.

        Args:
            value (int): The value to set.
        """
        ...
    def getOutputHeight(self) -> int:
        """
        Gets the value for outputheight.

        Returns:
            int: The current value.
        """
        ...
    def setIn_stereo(self, value: int) -> None:
        """
        Sets the value for in_stereo.

        Args:
            value (int): The value to set.
        """
        ...
    def getIn_stereo(self) -> int:
        """
        Gets the value for in_stereo.

        Returns:
            int: The current value.
        """
        ...
    def setOut_stereo(self, value: int) -> None:
        """
        Sets the value for out_stereo.

        Args:
            value (int): The value to set.
        """
        ...
    def getOut_stereo(self) -> int:
        """
        Gets the value for out_stereo.

        Returns:
            int: The current value.
        """
        ...
    def setIn_forder(self, value: Any) -> None:
        """
        Sets the value for in_forder.

        Args:
            value (Any): The value to set.
        """
        ...
    def getIn_forder(self) -> Any:
        """
        Gets the value for in_forder.

        Returns:
            Any: The current value.
        """
        ...
    def setOut_forder(self, value: Any) -> None:
        """
        Sets the value for out_forder.

        Args:
            value (Any): The value to set.
        """
        ...
    def getOut_forder(self) -> Any:
        """
        Gets the value for out_forder.

        Returns:
            Any: The current value.
        """
        ...
    def setIn_frot(self, value: Any) -> None:
        """
        Sets the value for in_frot.

        Args:
            value (Any): The value to set.
        """
        ...
    def getIn_frot(self) -> Any:
        """
        Gets the value for in_frot.

        Returns:
            Any: The current value.
        """
        ...
    def setOut_frot(self, value: Any) -> None:
        """
        Sets the value for out_frot.

        Args:
            value (Any): The value to set.
        """
        ...
    def getOut_frot(self) -> Any:
        """
        Gets the value for out_frot.

        Returns:
            Any: The current value.
        """
        ...
    def setIn_pad(self, value: float) -> None:
        """
        Sets the value for in_pad.

        Args:
            value (float): The value to set.
        """
        ...
    def getIn_pad(self) -> float:
        """
        Gets the value for in_pad.

        Returns:
            float: The current value.
        """
        ...
    def setOut_pad(self, value: float) -> None:
        """
        Sets the value for out_pad.

        Args:
            value (float): The value to set.
        """
        ...
    def getOut_pad(self) -> float:
        """
        Gets the value for out_pad.

        Returns:
            float: The current value.
        """
        ...
    def setFin_pad(self, value: int) -> None:
        """
        Sets the value for fin_pad.

        Args:
            value (int): The value to set.
        """
        ...
    def getFin_pad(self) -> int:
        """
        Gets the value for fin_pad.

        Returns:
            int: The current value.
        """
        ...
    def setFout_pad(self, value: int) -> None:
        """
        Sets the value for fout_pad.

        Args:
            value (int): The value to set.
        """
        ...
    def getFout_pad(self) -> int:
        """
        Gets the value for fout_pad.

        Returns:
            int: The current value.
        """
        ...
    def setYaw(self, value: float) -> None:
        """
        Sets the value for yaw.

        Args:
            value (float): The value to set.
        """
        ...
    def getYaw(self) -> float:
        """
        Gets the value for yaw.

        Returns:
            float: The current value.
        """
        ...
    def setPitch(self, value: float) -> None:
        """
        Sets the value for pitch.

        Args:
            value (float): The value to set.
        """
        ...
    def getPitch(self) -> float:
        """
        Gets the value for pitch.

        Returns:
            float: The current value.
        """
        ...
    def setRoll(self, value: float) -> None:
        """
        Sets the value for roll.

        Args:
            value (float): The value to set.
        """
        ...
    def getRoll(self) -> float:
        """
        Gets the value for roll.

        Returns:
            float: The current value.
        """
        ...
    def setRorder(self, value: Any) -> None:
        """
        Sets the value for rorder.

        Args:
            value (Any): The value to set.
        """
        ...
    def getRorder(self) -> Any:
        """
        Gets the value for rorder.

        Returns:
            Any: The current value.
        """
        ...
    def setH_fov(self, value: float) -> None:
        """
        Sets the value for h_fov.

        Args:
            value (float): The value to set.
        """
        ...
    def getH_fov(self) -> float:
        """
        Gets the value for h_fov.

        Returns:
            float: The current value.
        """
        ...
    def setV_fov(self, value: float) -> None:
        """
        Sets the value for v_fov.

        Args:
            value (float): The value to set.
        """
        ...
    def getV_fov(self) -> float:
        """
        Gets the value for v_fov.

        Returns:
            float: The current value.
        """
        ...
    def setD_fov(self, value: float) -> None:
        """
        Sets the value for d_fov.

        Args:
            value (float): The value to set.
        """
        ...
    def getD_fov(self) -> float:
        """
        Gets the value for d_fov.

        Returns:
            float: The current value.
        """
        ...
    def setH_flip(self, value: bool) -> None:
        """
        Sets the value for h_flip.

        Args:
            value (bool): The value to set.
        """
        ...
    def getH_flip(self) -> bool:
        """
        Gets the value for h_flip.

        Returns:
            bool: The current value.
        """
        ...
    def setV_flip(self, value: bool) -> None:
        """
        Sets the value for v_flip.

        Args:
            value (bool): The value to set.
        """
        ...
    def getV_flip(self) -> bool:
        """
        Gets the value for v_flip.

        Returns:
            bool: The current value.
        """
        ...
    def setD_flip(self, value: bool) -> None:
        """
        Sets the value for d_flip.

        Args:
            value (bool): The value to set.
        """
        ...
    def getD_flip(self) -> bool:
        """
        Gets the value for d_flip.

        Returns:
            bool: The current value.
        """
        ...
    def setIh_flip(self, value: bool) -> None:
        """
        Sets the value for ih_flip.

        Args:
            value (bool): The value to set.
        """
        ...
    def getIh_flip(self) -> bool:
        """
        Gets the value for ih_flip.

        Returns:
            bool: The current value.
        """
        ...
    def setIv_flip(self, value: bool) -> None:
        """
        Sets the value for iv_flip.

        Args:
            value (bool): The value to set.
        """
        ...
    def getIv_flip(self) -> bool:
        """
        Gets the value for iv_flip.

        Returns:
            bool: The current value.
        """
        ...
    def setIn_trans(self, value: bool) -> None:
        """
        Sets the value for in_trans.

        Args:
            value (bool): The value to set.
        """
        ...
    def getIn_trans(self) -> bool:
        """
        Gets the value for in_trans.

        Returns:
            bool: The current value.
        """
        ...
    def setOut_trans(self, value: bool) -> None:
        """
        Sets the value for out_trans.

        Args:
            value (bool): The value to set.
        """
        ...
    def getOut_trans(self) -> bool:
        """
        Gets the value for out_trans.

        Returns:
            bool: The current value.
        """
        ...
    def setIh_fov(self, value: float) -> None:
        """
        Sets the value for ih_fov.

        Args:
            value (float): The value to set.
        """
        ...
    def getIh_fov(self) -> float:
        """
        Gets the value for ih_fov.

        Returns:
            float: The current value.
        """
        ...
    def setIv_fov(self, value: float) -> None:
        """
        Sets the value for iv_fov.

        Args:
            value (float): The value to set.
        """
        ...
    def getIv_fov(self) -> float:
        """
        Gets the value for iv_fov.

        Returns:
            float: The current value.
        """
        ...
    def setId_fov(self, value: float) -> None:
        """
        Sets the value for id_fov.

        Args:
            value (float): The value to set.
        """
        ...
    def getId_fov(self) -> float:
        """
        Gets the value for id_fov.

        Returns:
            float: The current value.
        """
        ...
    def setH_offset(self, value: float) -> None:
        """
        Sets the value for h_offset.

        Args:
            value (float): The value to set.
        """
        ...
    def getH_offset(self) -> float:
        """
        Gets the value for h_offset.

        Returns:
            float: The current value.
        """
        ...
    def setV_offset(self, value: float) -> None:
        """
        Sets the value for v_offset.

        Args:
            value (float): The value to set.
        """
        ...
    def getV_offset(self) -> float:
        """
        Gets the value for v_offset.

        Returns:
            float: The current value.
        """
        ...
    def setAlpha_mask(self, value: bool) -> None:
        """
        Sets the value for alpha_mask.

        Args:
            value (bool): The value to set.
        """
        ...
    def getAlpha_mask(self) -> bool:
        """
        Gets the value for alpha_mask.

        Returns:
            bool: The current value.
        """
        ...
    def setReset_rot(self, value: bool) -> None:
        """
        Sets the value for reset_rot.

        Args:
            value (bool): The value to set.
        """
        ...
    def getReset_rot(self) -> bool:
        """
        Gets the value for reset_rot.

        Returns:
            bool: The current value.
        """
        ...

class Varblur(FilterBase):
    """
    Apply Variable Blur filter.
    """
    def setMin_r(self, value: int) -> None:
        """
        Sets the value for min_r.

        Args:
            value (int): The value to set.
        """
        ...
    def getMin_r(self) -> int:
        """
        Gets the value for min_r.

        Returns:
            int: The current value.
        """
        ...
    def setMax_r(self, value: int) -> None:
        """
        Sets the value for max_r.

        Args:
            value (int): The value to set.
        """
        ...
    def getMax_r(self) -> int:
        """
        Gets the value for max_r.

        Returns:
            int: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...

class Vectorscope(FilterBase):
    """
    Video vectorscope.
    """
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setColorComponentOnXAxis(self, value: int) -> None:
        """
        Sets the value for colorcomponentonxaxis.

        Args:
            value (int): The value to set.
        """
        ...
    def getColorComponentOnXAxis(self) -> int:
        """
        Gets the value for colorcomponentonxaxis.

        Returns:
            int: The current value.
        """
        ...
    def setColorComponentOnYAxis(self, value: int) -> None:
        """
        Sets the value for colorcomponentonyaxis.

        Args:
            value (int): The value to set.
        """
        ...
    def getColorComponentOnYAxis(self) -> int:
        """
        Gets the value for colorcomponentonyaxis.

        Returns:
            int: The current value.
        """
        ...
    def setIntensity(self, value: float) -> None:
        """
        Sets the value for intensity.

        Args:
            value (float): The value to set.
        """
        ...
    def getIntensity(self) -> float:
        """
        Gets the value for intensity.

        Returns:
            float: The current value.
        """
        ...
    def setEnvelope(self, value: int) -> None:
        """
        Sets the value for envelope.

        Args:
            value (int): The value to set.
        """
        ...
    def getEnvelope(self) -> int:
        """
        Gets the value for envelope.

        Returns:
            int: The current value.
        """
        ...
    def setGraticule(self, value: int) -> None:
        """
        Sets the value for graticule.

        Args:
            value (int): The value to set.
        """
        ...
    def getGraticule(self) -> int:
        """
        Gets the value for graticule.

        Returns:
            int: The current value.
        """
        ...
    def setOpacity(self, value: float) -> None:
        """
        Sets the value for opacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getOpacity(self) -> float:
        """
        Gets the value for opacity.

        Returns:
            float: The current value.
        """
        ...
    def setFlags(self, value: int) -> None:
        """
        Sets the value for flags.

        Args:
            value (int): The value to set.
        """
        ...
    def getFlags(self) -> int:
        """
        Gets the value for flags.

        Returns:
            int: The current value.
        """
        ...
    def setBgopacity(self, value: float) -> None:
        """
        Sets the value for bgopacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getBgopacity(self) -> float:
        """
        Gets the value for bgopacity.

        Returns:
            float: The current value.
        """
        ...
    def setLthreshold(self, value: float) -> None:
        """
        Sets the value for lthreshold.

        Args:
            value (float): The value to set.
        """
        ...
    def getLthreshold(self) -> float:
        """
        Gets the value for lthreshold.

        Returns:
            float: The current value.
        """
        ...
    def setHthreshold(self, value: float) -> None:
        """
        Sets the value for hthreshold.

        Args:
            value (float): The value to set.
        """
        ...
    def getHthreshold(self) -> float:
        """
        Gets the value for hthreshold.

        Returns:
            float: The current value.
        """
        ...
    def setColorspace(self, value: int) -> None:
        """
        Sets the value for colorspace.

        Args:
            value (int): The value to set.
        """
        ...
    def getColorspace(self) -> int:
        """
        Gets the value for colorspace.

        Returns:
            int: The current value.
        """
        ...
    def setTint0(self, value: float) -> None:
        """
        Sets the value for tint0.

        Args:
            value (float): The value to set.
        """
        ...
    def getTint0(self) -> float:
        """
        Gets the value for tint0.

        Returns:
            float: The current value.
        """
        ...
    def setTint1(self, value: float) -> None:
        """
        Sets the value for tint1.

        Args:
            value (float): The value to set.
        """
        ...
    def getTint1(self) -> float:
        """
        Gets the value for tint1.

        Returns:
            float: The current value.
        """
        ...

class Vflip(FilterBase):
    """
    Flip the input video vertically.
    """
    pass

class Vfrdet(FilterBase):
    """
    Variable frame rate detect filter.
    """
    pass

class Vibrance(FilterBase):
    """
    Boost or alter saturation.
    """
    def setIntensity(self, value: float) -> None:
        """
        Sets the value for intensity.

        Args:
            value (float): The value to set.
        """
        ...
    def getIntensity(self) -> float:
        """
        Gets the value for intensity.

        Returns:
            float: The current value.
        """
        ...
    def setRbal(self, value: float) -> None:
        """
        Sets the value for rbal.

        Args:
            value (float): The value to set.
        """
        ...
    def getRbal(self) -> float:
        """
        Gets the value for rbal.

        Returns:
            float: The current value.
        """
        ...
    def setGbal(self, value: float) -> None:
        """
        Sets the value for gbal.

        Args:
            value (float): The value to set.
        """
        ...
    def getGbal(self) -> float:
        """
        Gets the value for gbal.

        Returns:
            float: The current value.
        """
        ...
    def setBbal(self, value: float) -> None:
        """
        Sets the value for bbal.

        Args:
            value (float): The value to set.
        """
        ...
    def getBbal(self) -> float:
        """
        Gets the value for bbal.

        Returns:
            float: The current value.
        """
        ...
    def setRlum(self, value: float) -> None:
        """
        Sets the value for rlum.

        Args:
            value (float): The value to set.
        """
        ...
    def getRlum(self) -> float:
        """
        Gets the value for rlum.

        Returns:
            float: The current value.
        """
        ...
    def setGlum(self, value: float) -> None:
        """
        Sets the value for glum.

        Args:
            value (float): The value to set.
        """
        ...
    def getGlum(self) -> float:
        """
        Gets the value for glum.

        Returns:
            float: The current value.
        """
        ...
    def setBlum(self, value: float) -> None:
        """
        Sets the value for blum.

        Args:
            value (float): The value to set.
        """
        ...
    def getBlum(self) -> float:
        """
        Gets the value for blum.

        Returns:
            float: The current value.
        """
        ...
    def setAlternate(self, value: bool) -> None:
        """
        Sets the value for alternate.

        Args:
            value (bool): The value to set.
        """
        ...
    def getAlternate(self) -> bool:
        """
        Gets the value for alternate.

        Returns:
            bool: The current value.
        """
        ...

class Vif(FilterBase):
    """
    Calculate the VIF between two video streams.
    """
    pass

class Vignette(FilterBase):
    """
    Make or reverse a vignette effect.
    """
    def setAngle(self, value: Any) -> None:
        """
        Sets the value for angle.

        Args:
            value (Any): The value to set.
        """
        ...
    def getAngle(self) -> Any:
        """
        Gets the value for angle.

        Returns:
            Any: The current value.
        """
        ...
    def setX0(self, value: Any) -> None:
        """
        Sets the value for x0.

        Args:
            value (Any): The value to set.
        """
        ...
    def getX0(self) -> Any:
        """
        Gets the value for x0.

        Returns:
            Any: The current value.
        """
        ...
    def setY0(self, value: Any) -> None:
        """
        Sets the value for y0.

        Args:
            value (Any): The value to set.
        """
        ...
    def getY0(self) -> Any:
        """
        Gets the value for y0.

        Returns:
            Any: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setEval(self, value: int) -> None:
        """
        Sets the value for eval.

        Args:
            value (int): The value to set.
        """
        ...
    def getEval(self) -> int:
        """
        Gets the value for eval.

        Returns:
            int: The current value.
        """
        ...
    def setDither(self, value: bool) -> None:
        """
        Sets the value for dither.

        Args:
            value (bool): The value to set.
        """
        ...
    def getDither(self) -> bool:
        """
        Gets the value for dither.

        Returns:
            bool: The current value.
        """
        ...
    def setAspect(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for aspect.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getAspect(self) -> Tuple[int, int]:
        """
        Gets the value for aspect.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...

class Vmafmotion(FilterBase):
    """
    Calculate the VMAF Motion score.
    """
    def setStats_file(self, value: Any) -> None:
        """
        Sets the value for stats_file.

        Args:
            value (Any): The value to set.
        """
        ...
    def getStats_file(self) -> Any:
        """
        Gets the value for stats_file.

        Returns:
            Any: The current value.
        """
        ...

class Vstack(FilterBase):
    """
    Stack video inputs vertically.
    """
    def setInputs(self, value: int) -> None:
        """
        Sets the value for inputs.

        Args:
            value (int): The value to set.
        """
        ...
    def getInputs(self) -> int:
        """
        Gets the value for inputs.

        Returns:
            int: The current value.
        """
        ...
    def setShortest(self, value: bool) -> None:
        """
        Sets the value for shortest.

        Args:
            value (bool): The value to set.
        """
        ...
    def getShortest(self) -> bool:
        """
        Gets the value for shortest.

        Returns:
            bool: The current value.
        """
        ...

class W3fdif(FilterBase):
    """
    Apply Martin Weston three field deinterlace.
    """
    def setFilter(self, value: int) -> None:
        """
        Sets the value for filter.

        Args:
            value (int): The value to set.
        """
        ...
    def getFilter(self) -> int:
        """
        Gets the value for filter.

        Returns:
            int: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setParity(self, value: int) -> None:
        """
        Sets the value for parity.

        Args:
            value (int): The value to set.
        """
        ...
    def getParity(self) -> int:
        """
        Gets the value for parity.

        Returns:
            int: The current value.
        """
        ...
    def setDeint(self, value: int) -> None:
        """
        Sets the value for deint.

        Args:
            value (int): The value to set.
        """
        ...
    def getDeint(self) -> int:
        """
        Gets the value for deint.

        Returns:
            int: The current value.
        """
        ...

class Waveform(FilterBase):
    """
    Video waveform monitor.
    """
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setIntensity(self, value: float) -> None:
        """
        Sets the value for intensity.

        Args:
            value (float): The value to set.
        """
        ...
    def getIntensity(self) -> float:
        """
        Gets the value for intensity.

        Returns:
            float: The current value.
        """
        ...
    def setMirror(self, value: bool) -> None:
        """
        Sets the value for mirror.

        Args:
            value (bool): The value to set.
        """
        ...
    def getMirror(self) -> bool:
        """
        Gets the value for mirror.

        Returns:
            bool: The current value.
        """
        ...
    def setDisplay(self, value: int) -> None:
        """
        Sets the value for display.

        Args:
            value (int): The value to set.
        """
        ...
    def getDisplay(self) -> int:
        """
        Gets the value for display.

        Returns:
            int: The current value.
        """
        ...
    def setComponents(self, value: int) -> None:
        """
        Sets the value for components.

        Args:
            value (int): The value to set.
        """
        ...
    def getComponents(self) -> int:
        """
        Gets the value for components.

        Returns:
            int: The current value.
        """
        ...
    def setEnvelope(self, value: int) -> None:
        """
        Sets the value for envelope.

        Args:
            value (int): The value to set.
        """
        ...
    def getEnvelope(self) -> int:
        """
        Gets the value for envelope.

        Returns:
            int: The current value.
        """
        ...
    def setFilter(self, value: int) -> None:
        """
        Sets the value for filter.

        Args:
            value (int): The value to set.
        """
        ...
    def getFilter(self) -> int:
        """
        Gets the value for filter.

        Returns:
            int: The current value.
        """
        ...
    def setGraticule(self, value: int) -> None:
        """
        Sets the value for graticule.

        Args:
            value (int): The value to set.
        """
        ...
    def getGraticule(self) -> int:
        """
        Gets the value for graticule.

        Returns:
            int: The current value.
        """
        ...
    def setOpacity(self, value: float) -> None:
        """
        Sets the value for opacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getOpacity(self) -> float:
        """
        Gets the value for opacity.

        Returns:
            float: The current value.
        """
        ...
    def setFlags(self, value: int) -> None:
        """
        Sets the value for flags.

        Args:
            value (int): The value to set.
        """
        ...
    def getFlags(self) -> int:
        """
        Gets the value for flags.

        Returns:
            int: The current value.
        """
        ...
    def setScale(self, value: int) -> None:
        """
        Sets the value for scale.

        Args:
            value (int): The value to set.
        """
        ...
    def getScale(self) -> int:
        """
        Gets the value for scale.

        Returns:
            int: The current value.
        """
        ...
    def setBgopacity(self, value: float) -> None:
        """
        Sets the value for bgopacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getBgopacity(self) -> float:
        """
        Gets the value for bgopacity.

        Returns:
            float: The current value.
        """
        ...
    def setTint0(self, value: float) -> None:
        """
        Sets the value for tint0.

        Args:
            value (float): The value to set.
        """
        ...
    def getTint0(self) -> float:
        """
        Gets the value for tint0.

        Returns:
            float: The current value.
        """
        ...
    def setTint1(self, value: float) -> None:
        """
        Sets the value for tint1.

        Args:
            value (float): The value to set.
        """
        ...
    def getTint1(self) -> float:
        """
        Gets the value for tint1.

        Returns:
            float: The current value.
        """
        ...
    def setFitmode(self, value: int) -> None:
        """
        Sets the value for fitmode.

        Args:
            value (int): The value to set.
        """
        ...
    def getFitmode(self) -> int:
        """
        Gets the value for fitmode.

        Returns:
            int: The current value.
        """
        ...
    def setInput(self, value: int) -> None:
        """
        Sets the value for input.

        Args:
            value (int): The value to set.
        """
        ...
    def getInput(self) -> int:
        """
        Gets the value for input.

        Returns:
            int: The current value.
        """
        ...

class Weave(FilterBase):
    """
    Weave input video fields into frames.
    """
    def setFirst_field(self, value: int) -> None:
        """
        Sets the value for first_field.

        Args:
            value (int): The value to set.
        """
        ...
    def getFirst_field(self) -> int:
        """
        Gets the value for first_field.

        Returns:
            int: The current value.
        """
        ...

class Xbr(FilterBase):
    """
    Scale the input using xBR algorithm.
    """
    def setScaleFactor(self, value: int) -> None:
        """
        Sets the value for scalefactor.

        Args:
            value (int): The value to set.
        """
        ...
    def getScaleFactor(self) -> int:
        """
        Gets the value for scalefactor.

        Returns:
            int: The current value.
        """
        ...

class Xcorrelate(FilterBase):
    """
    Cross-correlate first video stream with second video stream.
    """
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setSecondary(self, value: int) -> None:
        """
        Sets the value for secondary.

        Args:
            value (int): The value to set.
        """
        ...
    def getSecondary(self) -> int:
        """
        Gets the value for secondary.

        Returns:
            int: The current value.
        """
        ...

class Xfade(FilterBase):
    """
    Cross fade one video with another video.
    """
    def setTransition(self, value: int) -> None:
        """
        Sets the value for transition.

        Args:
            value (int): The value to set.
        """
        ...
    def getTransition(self) -> int:
        """
        Gets the value for transition.

        Returns:
            int: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...
    def setOffset(self, value: int) -> None:
        """
        Sets the value for offset.

        Args:
            value (int): The value to set.
        """
        ...
    def getOffset(self) -> int:
        """
        Gets the value for offset.

        Returns:
            int: The current value.
        """
        ...
    def setExpr(self, value: Any) -> None:
        """
        Sets the value for expr.

        Args:
            value (Any): The value to set.
        """
        ...
    def getExpr(self) -> Any:
        """
        Gets the value for expr.

        Returns:
            Any: The current value.
        """
        ...

class Xmedian(FilterBase):
    """
    Pick median pixels from several video inputs.
    """
    def setInputs(self, value: int) -> None:
        """
        Sets the value for inputs.

        Args:
            value (int): The value to set.
        """
        ...
    def getInputs(self) -> int:
        """
        Gets the value for inputs.

        Returns:
            int: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setPercentile(self, value: float) -> None:
        """
        Sets the value for percentile.

        Args:
            value (float): The value to set.
        """
        ...
    def getPercentile(self) -> float:
        """
        Gets the value for percentile.

        Returns:
            float: The current value.
        """
        ...

class Xstack(FilterBase):
    """
    Stack video inputs into custom layout.
    """
    def setInputs(self, value: int) -> None:
        """
        Sets the value for inputs.

        Args:
            value (int): The value to set.
        """
        ...
    def getInputs(self) -> int:
        """
        Gets the value for inputs.

        Returns:
            int: The current value.
        """
        ...
    def setLayout(self, value: Any) -> None:
        """
        Sets the value for layout.

        Args:
            value (Any): The value to set.
        """
        ...
    def getLayout(self) -> Any:
        """
        Gets the value for layout.

        Returns:
            Any: The current value.
        """
        ...
    def setGrid(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for grid.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getGrid(self) -> Tuple[int, int]:
        """
        Gets the value for grid.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setShortest(self, value: bool) -> None:
        """
        Sets the value for shortest.

        Args:
            value (bool): The value to set.
        """
        ...
    def getShortest(self) -> bool:
        """
        Gets the value for shortest.

        Returns:
            bool: The current value.
        """
        ...
    def setFill(self, value: Any) -> None:
        """
        Sets the value for fill.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFill(self) -> Any:
        """
        Gets the value for fill.

        Returns:
            Any: The current value.
        """
        ...

class Yadif(FilterBase):
    """
    Deinterlace the input image.
    """
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setParity(self, value: int) -> None:
        """
        Sets the value for parity.

        Args:
            value (int): The value to set.
        """
        ...
    def getParity(self) -> int:
        """
        Gets the value for parity.

        Returns:
            int: The current value.
        """
        ...
    def setDeint(self, value: int) -> None:
        """
        Sets the value for deint.

        Args:
            value (int): The value to set.
        """
        ...
    def getDeint(self) -> int:
        """
        Gets the value for deint.

        Returns:
            int: The current value.
        """
        ...

class Yaepblur(FilterBase):
    """
    Yet another edge preserving blur filter.
    """
    def setRadius(self, value: int) -> None:
        """
        Sets the value for radius.

        Args:
            value (int): The value to set.
        """
        ...
    def getRadius(self) -> int:
        """
        Gets the value for radius.

        Returns:
            int: The current value.
        """
        ...
    def setPlanes(self, value: int) -> None:
        """
        Sets the value for planes.

        Args:
            value (int): The value to set.
        """
        ...
    def getPlanes(self) -> int:
        """
        Gets the value for planes.

        Returns:
            int: The current value.
        """
        ...
    def setSigma(self, value: int) -> None:
        """
        Sets the value for sigma.

        Args:
            value (int): The value to set.
        """
        ...
    def getSigma(self) -> int:
        """
        Gets the value for sigma.

        Returns:
            int: The current value.
        """
        ...

class Zoompan(FilterBase):
    """
    Apply Zoom & Pan effect.
    """
    def setZoom(self, value: Any) -> None:
        """
        Sets the value for zoom.

        Args:
            value (Any): The value to set.
        """
        ...
    def getZoom(self) -> Any:
        """
        Gets the value for zoom.

        Returns:
            Any: The current value.
        """
        ...
    def setX(self, value: Any) -> None:
        """
        Sets the value for x.

        Args:
            value (Any): The value to set.
        """
        ...
    def getX(self) -> Any:
        """
        Gets the value for x.

        Returns:
            Any: The current value.
        """
        ...
    def setY(self, value: Any) -> None:
        """
        Sets the value for y.

        Args:
            value (Any): The value to set.
        """
        ...
    def getY(self) -> Any:
        """
        Gets the value for y.

        Returns:
            Any: The current value.
        """
        ...
    def setDuration(self, value: Any) -> None:
        """
        Sets the value for duration.

        Args:
            value (Any): The value to set.
        """
        ...
    def getDuration(self) -> Any:
        """
        Gets the value for duration.

        Returns:
            Any: The current value.
        """
        ...
    def setOutputImageSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for outputimagesize.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getOutputImageSize(self) -> Tuple[int, int]:
        """
        Gets the value for outputimagesize.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setFps(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for fps.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getFps(self) -> Tuple[int, int]:
        """
        Gets the value for fps.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...

class Allrgb(FilterBase):
    """
    Generate all RGB colors.
    """
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...
    def setSar(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for sar.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSar(self) -> Tuple[int, int]:
        """
        Gets the value for sar.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...

class Allyuv(FilterBase):
    """
    Generate all yuv colors.
    """
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...
    def setSar(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for sar.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSar(self) -> Tuple[int, int]:
        """
        Gets the value for sar.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...

class Cellauto(FilterBase):
    """
    Create pattern generated by an elementary cellular automaton.
    """
    def setFilename(self, value: Any) -> None:
        """
        Sets the value for filename.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFilename(self) -> Any:
        """
        Gets the value for filename.

        Returns:
            Any: The current value.
        """
        ...
    def setPattern(self, value: Any) -> None:
        """
        Sets the value for pattern.

        Args:
            value (Any): The value to set.
        """
        ...
    def getPattern(self) -> Any:
        """
        Gets the value for pattern.

        Returns:
            Any: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRule(self, value: int) -> None:
        """
        Sets the value for rule.

        Args:
            value (int): The value to set.
        """
        ...
    def getRule(self) -> int:
        """
        Gets the value for rule.

        Returns:
            int: The current value.
        """
        ...
    def setRandom_fill_ratio(self, value: float) -> None:
        """
        Sets the value for random_fill_ratio.

        Args:
            value (float): The value to set.
        """
        ...
    def getRandom_fill_ratio(self) -> float:
        """
        Gets the value for random_fill_ratio.

        Returns:
            float: The current value.
        """
        ...
    def setRandom_seed(self, value: int) -> None:
        """
        Sets the value for random_seed.

        Args:
            value (int): The value to set.
        """
        ...
    def getRandom_seed(self) -> int:
        """
        Gets the value for random_seed.

        Returns:
            int: The current value.
        """
        ...
    def setScroll(self, value: bool) -> None:
        """
        Sets the value for scroll.

        Args:
            value (bool): The value to set.
        """
        ...
    def getScroll(self) -> bool:
        """
        Gets the value for scroll.

        Returns:
            bool: The current value.
        """
        ...
    def setStart_full(self, value: bool) -> None:
        """
        Sets the value for start_full.

        Args:
            value (bool): The value to set.
        """
        ...
    def getStart_full(self) -> bool:
        """
        Gets the value for start_full.

        Returns:
            bool: The current value.
        """
        ...
    def setFull(self, value: bool) -> None:
        """
        Sets the value for full.

        Args:
            value (bool): The value to set.
        """
        ...
    def getFull(self) -> bool:
        """
        Gets the value for full.

        Returns:
            bool: The current value.
        """
        ...
    def setStitch(self, value: bool) -> None:
        """
        Sets the value for stitch.

        Args:
            value (bool): The value to set.
        """
        ...
    def getStitch(self) -> bool:
        """
        Gets the value for stitch.

        Returns:
            bool: The current value.
        """
        ...

class Color(FilterBase):
    """
    Provide an uniformly colored input.
    """
    def setColor(self, value: Any) -> None:
        """
        Sets the value for color.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColor(self) -> Any:
        """
        Gets the value for color.

        Returns:
            Any: The current value.
        """
        ...
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...
    def setSar(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for sar.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSar(self) -> Tuple[int, int]:
        """
        Gets the value for sar.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...

class Colorchart(FilterBase):
    """
    Generate color checker chart.
    """
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...
    def setSar(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for sar.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSar(self) -> Tuple[int, int]:
        """
        Gets the value for sar.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setPatch_size(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for patch_size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getPatch_size(self) -> Tuple[int, int]:
        """
        Gets the value for patch_size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setPreset(self, value: int) -> None:
        """
        Sets the value for preset.

        Args:
            value (int): The value to set.
        """
        ...
    def getPreset(self) -> int:
        """
        Gets the value for preset.

        Returns:
            int: The current value.
        """
        ...

class Colorspectrum(FilterBase):
    """
    Generate colors spectrum.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...
    def setSar(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for sar.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSar(self) -> Tuple[int, int]:
        """
        Gets the value for sar.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setType(self, value: int) -> None:
        """
        Sets the value for type.

        Args:
            value (int): The value to set.
        """
        ...
    def getType(self) -> int:
        """
        Gets the value for type.

        Returns:
            int: The current value.
        """
        ...

class Ddagrab(FilterBase):
    """
    Grab Windows Desktop images using Desktop Duplication API
    """
    def setOutput_idx(self, value: int) -> None:
        """
        Sets the value for output_idx.

        Args:
            value (int): The value to set.
        """
        ...
    def getOutput_idx(self) -> int:
        """
        Gets the value for output_idx.

        Returns:
            int: The current value.
        """
        ...
    def setDraw_mouse(self, value: bool) -> None:
        """
        Sets the value for draw_mouse.

        Args:
            value (bool): The value to set.
        """
        ...
    def getDraw_mouse(self) -> bool:
        """
        Gets the value for draw_mouse.

        Returns:
            bool: The current value.
        """
        ...
    def setFramerate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for framerate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getFramerate(self) -> Tuple[int, int]:
        """
        Gets the value for framerate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setVideo_size(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for video_size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getVideo_size(self) -> Tuple[int, int]:
        """
        Gets the value for video_size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setOffset_x(self, value: int) -> None:
        """
        Sets the value for offset_x.

        Args:
            value (int): The value to set.
        """
        ...
    def getOffset_x(self) -> int:
        """
        Gets the value for offset_x.

        Returns:
            int: The current value.
        """
        ...
    def setOffset_y(self, value: int) -> None:
        """
        Sets the value for offset_y.

        Args:
            value (int): The value to set.
        """
        ...
    def getOffset_y(self) -> int:
        """
        Gets the value for offset_y.

        Returns:
            int: The current value.
        """
        ...
    def setOutput_fmt(self, value: int) -> None:
        """
        Sets the value for output_fmt.

        Args:
            value (int): The value to set.
        """
        ...
    def getOutput_fmt(self) -> int:
        """
        Gets the value for output_fmt.

        Returns:
            int: The current value.
        """
        ...
    def setAllow_fallback(self, value: bool) -> None:
        """
        Sets the value for allow_fallback.

        Args:
            value (bool): The value to set.
        """
        ...
    def getAllow_fallback(self) -> bool:
        """
        Gets the value for allow_fallback.

        Returns:
            bool: The current value.
        """
        ...
    def setForce_fmt(self, value: bool) -> None:
        """
        Sets the value for force_fmt.

        Args:
            value (bool): The value to set.
        """
        ...
    def getForce_fmt(self) -> bool:
        """
        Gets the value for force_fmt.

        Returns:
            bool: The current value.
        """
        ...
    def setDup_frames(self, value: bool) -> None:
        """
        Sets the value for dup_frames.

        Args:
            value (bool): The value to set.
        """
        ...
    def getDup_frames(self) -> bool:
        """
        Gets the value for dup_frames.

        Returns:
            bool: The current value.
        """
        ...

class Gradients(FilterBase):
    """
    Draw a gradients.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setC0(self, value: Any) -> None:
        """
        Sets the value for c0.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC0(self) -> Any:
        """
        Gets the value for c0.

        Returns:
            Any: The current value.
        """
        ...
    def setC1(self, value: Any) -> None:
        """
        Sets the value for c1.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC1(self) -> Any:
        """
        Gets the value for c1.

        Returns:
            Any: The current value.
        """
        ...
    def setC2(self, value: Any) -> None:
        """
        Sets the value for c2.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC2(self) -> Any:
        """
        Gets the value for c2.

        Returns:
            Any: The current value.
        """
        ...
    def setC3(self, value: Any) -> None:
        """
        Sets the value for c3.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC3(self) -> Any:
        """
        Gets the value for c3.

        Returns:
            Any: The current value.
        """
        ...
    def setC4(self, value: Any) -> None:
        """
        Sets the value for c4.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC4(self) -> Any:
        """
        Gets the value for c4.

        Returns:
            Any: The current value.
        """
        ...
    def setC5(self, value: Any) -> None:
        """
        Sets the value for c5.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC5(self) -> Any:
        """
        Gets the value for c5.

        Returns:
            Any: The current value.
        """
        ...
    def setC6(self, value: Any) -> None:
        """
        Sets the value for c6.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC6(self) -> Any:
        """
        Gets the value for c6.

        Returns:
            Any: The current value.
        """
        ...
    def setC7(self, value: Any) -> None:
        """
        Sets the value for c7.

        Args:
            value (Any): The value to set.
        """
        ...
    def getC7(self) -> Any:
        """
        Gets the value for c7.

        Returns:
            Any: The current value.
        """
        ...
    def setX0(self, value: int) -> None:
        """
        Sets the value for x0.

        Args:
            value (int): The value to set.
        """
        ...
    def getX0(self) -> int:
        """
        Gets the value for x0.

        Returns:
            int: The current value.
        """
        ...
    def setY0(self, value: int) -> None:
        """
        Sets the value for y0.

        Args:
            value (int): The value to set.
        """
        ...
    def getY0(self) -> int:
        """
        Gets the value for y0.

        Returns:
            int: The current value.
        """
        ...
    def setX1(self, value: int) -> None:
        """
        Sets the value for x1.

        Args:
            value (int): The value to set.
        """
        ...
    def getX1(self) -> int:
        """
        Gets the value for x1.

        Returns:
            int: The current value.
        """
        ...
    def setY1(self, value: int) -> None:
        """
        Sets the value for y1.

        Args:
            value (int): The value to set.
        """
        ...
    def getY1(self) -> int:
        """
        Gets the value for y1.

        Returns:
            int: The current value.
        """
        ...
    def setNb_colors(self, value: int) -> None:
        """
        Sets the value for nb_colors.

        Args:
            value (int): The value to set.
        """
        ...
    def getNb_colors(self) -> int:
        """
        Gets the value for nb_colors.

        Returns:
            int: The current value.
        """
        ...
    def setSeed(self, value: int) -> None:
        """
        Sets the value for seed.

        Args:
            value (int): The value to set.
        """
        ...
    def getSeed(self) -> int:
        """
        Gets the value for seed.

        Returns:
            int: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...
    def setSpeed(self, value: float) -> None:
        """
        Sets the value for speed.

        Args:
            value (float): The value to set.
        """
        ...
    def getSpeed(self) -> float:
        """
        Gets the value for speed.

        Returns:
            float: The current value.
        """
        ...
    def setType(self, value: int) -> None:
        """
        Sets the value for type.

        Args:
            value (int): The value to set.
        """
        ...
    def getType(self) -> int:
        """
        Gets the value for type.

        Returns:
            int: The current value.
        """
        ...

class Haldclutsrc(FilterBase):
    """
    Provide an identity Hald CLUT.
    """
    def setLevel(self, value: int) -> None:
        """
        Sets the value for level.

        Args:
            value (int): The value to set.
        """
        ...
    def getLevel(self) -> int:
        """
        Gets the value for level.

        Returns:
            int: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...
    def setSar(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for sar.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSar(self) -> Tuple[int, int]:
        """
        Gets the value for sar.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...

class Life(FilterBase):
    """
    Create life.
    """
    def setFilename(self, value: Any) -> None:
        """
        Sets the value for filename.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFilename(self) -> Any:
        """
        Gets the value for filename.

        Returns:
            Any: The current value.
        """
        ...
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRule(self, value: Any) -> None:
        """
        Sets the value for rule.

        Args:
            value (Any): The value to set.
        """
        ...
    def getRule(self) -> Any:
        """
        Gets the value for rule.

        Returns:
            Any: The current value.
        """
        ...
    def setRandom_fill_ratio(self, value: float) -> None:
        """
        Sets the value for random_fill_ratio.

        Args:
            value (float): The value to set.
        """
        ...
    def getRandom_fill_ratio(self) -> float:
        """
        Gets the value for random_fill_ratio.

        Returns:
            float: The current value.
        """
        ...
    def setRandom_seed(self, value: int) -> None:
        """
        Sets the value for random_seed.

        Args:
            value (int): The value to set.
        """
        ...
    def getRandom_seed(self) -> int:
        """
        Gets the value for random_seed.

        Returns:
            int: The current value.
        """
        ...
    def setStitch(self, value: bool) -> None:
        """
        Sets the value for stitch.

        Args:
            value (bool): The value to set.
        """
        ...
    def getStitch(self) -> bool:
        """
        Gets the value for stitch.

        Returns:
            bool: The current value.
        """
        ...
    def setMold(self, value: int) -> None:
        """
        Sets the value for mold.

        Args:
            value (int): The value to set.
        """
        ...
    def getMold(self) -> int:
        """
        Gets the value for mold.

        Returns:
            int: The current value.
        """
        ...
    def setLife_color(self, value: Any) -> None:
        """
        Sets the value for life_color.

        Args:
            value (Any): The value to set.
        """
        ...
    def getLife_color(self) -> Any:
        """
        Gets the value for life_color.

        Returns:
            Any: The current value.
        """
        ...
    def setDeath_color(self, value: Any) -> None:
        """
        Sets the value for death_color.

        Args:
            value (Any): The value to set.
        """
        ...
    def getDeath_color(self) -> Any:
        """
        Gets the value for death_color.

        Returns:
            Any: The current value.
        """
        ...
    def setMold_color(self, value: Any) -> None:
        """
        Sets the value for mold_color.

        Args:
            value (Any): The value to set.
        """
        ...
    def getMold_color(self) -> Any:
        """
        Gets the value for mold_color.

        Returns:
            Any: The current value.
        """
        ...

class Mandelbrot(FilterBase):
    """
    Render a Mandelbrot fractal.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setMaxiter(self, value: int) -> None:
        """
        Sets the value for maxiter.

        Args:
            value (int): The value to set.
        """
        ...
    def getMaxiter(self) -> int:
        """
        Gets the value for maxiter.

        Returns:
            int: The current value.
        """
        ...
    def setStart_x(self, value: float) -> None:
        """
        Sets the value for start_x.

        Args:
            value (float): The value to set.
        """
        ...
    def getStart_x(self) -> float:
        """
        Gets the value for start_x.

        Returns:
            float: The current value.
        """
        ...
    def setStart_y(self, value: float) -> None:
        """
        Sets the value for start_y.

        Args:
            value (float): The value to set.
        """
        ...
    def getStart_y(self) -> float:
        """
        Gets the value for start_y.

        Returns:
            float: The current value.
        """
        ...
    def setStart_scale(self, value: float) -> None:
        """
        Sets the value for start_scale.

        Args:
            value (float): The value to set.
        """
        ...
    def getStart_scale(self) -> float:
        """
        Gets the value for start_scale.

        Returns:
            float: The current value.
        """
        ...
    def setEnd_scale(self, value: float) -> None:
        """
        Sets the value for end_scale.

        Args:
            value (float): The value to set.
        """
        ...
    def getEnd_scale(self) -> float:
        """
        Gets the value for end_scale.

        Returns:
            float: The current value.
        """
        ...
    def setEnd_pts(self, value: float) -> None:
        """
        Sets the value for end_pts.

        Args:
            value (float): The value to set.
        """
        ...
    def getEnd_pts(self) -> float:
        """
        Gets the value for end_pts.

        Returns:
            float: The current value.
        """
        ...
    def setBailout(self, value: float) -> None:
        """
        Sets the value for bailout.

        Args:
            value (float): The value to set.
        """
        ...
    def getBailout(self) -> float:
        """
        Gets the value for bailout.

        Returns:
            float: The current value.
        """
        ...
    def setMorphxf(self, value: float) -> None:
        """
        Sets the value for morphxf.

        Args:
            value (float): The value to set.
        """
        ...
    def getMorphxf(self) -> float:
        """
        Gets the value for morphxf.

        Returns:
            float: The current value.
        """
        ...
    def setMorphyf(self, value: float) -> None:
        """
        Sets the value for morphyf.

        Args:
            value (float): The value to set.
        """
        ...
    def getMorphyf(self) -> float:
        """
        Gets the value for morphyf.

        Returns:
            float: The current value.
        """
        ...
    def setMorphamp(self, value: float) -> None:
        """
        Sets the value for morphamp.

        Args:
            value (float): The value to set.
        """
        ...
    def getMorphamp(self) -> float:
        """
        Gets the value for morphamp.

        Returns:
            float: The current value.
        """
        ...
    def setOuter(self, value: int) -> None:
        """
        Sets the value for outer.

        Args:
            value (int): The value to set.
        """
        ...
    def getOuter(self) -> int:
        """
        Gets the value for outer.

        Returns:
            int: The current value.
        """
        ...
    def setInner(self, value: int) -> None:
        """
        Sets the value for inner.

        Args:
            value (int): The value to set.
        """
        ...
    def getInner(self) -> int:
        """
        Gets the value for inner.

        Returns:
            int: The current value.
        """
        ...

class Nullsrc(FilterBase):
    """
    Null video source, return unprocessed video frames.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...
    def setSar(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for sar.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSar(self) -> Tuple[int, int]:
        """
        Gets the value for sar.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...

class Pal75bars(FilterBase):
    """
    Generate PAL 75% color bars.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...
    def setSar(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for sar.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSar(self) -> Tuple[int, int]:
        """
        Gets the value for sar.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...

class Pal100bars(FilterBase):
    """
    Generate PAL 100% color bars.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...
    def setSar(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for sar.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSar(self) -> Tuple[int, int]:
        """
        Gets the value for sar.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...

class Rgbtestsrc(FilterBase):
    """
    Generate RGB test pattern.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...
    def setSar(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for sar.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSar(self) -> Tuple[int, int]:
        """
        Gets the value for sar.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setComplement(self, value: bool) -> None:
        """
        Sets the value for complement.

        Args:
            value (bool): The value to set.
        """
        ...
    def getComplement(self) -> bool:
        """
        Gets the value for complement.

        Returns:
            bool: The current value.
        """
        ...

class Sierpinski(FilterBase):
    """
    Render a Sierpinski fractal.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setSeed(self, value: int) -> None:
        """
        Sets the value for seed.

        Args:
            value (int): The value to set.
        """
        ...
    def getSeed(self) -> int:
        """
        Gets the value for seed.

        Returns:
            int: The current value.
        """
        ...
    def setJump(self, value: int) -> None:
        """
        Sets the value for jump.

        Args:
            value (int): The value to set.
        """
        ...
    def getJump(self) -> int:
        """
        Gets the value for jump.

        Returns:
            int: The current value.
        """
        ...
    def setType(self, value: int) -> None:
        """
        Sets the value for type.

        Args:
            value (int): The value to set.
        """
        ...
    def getType(self) -> int:
        """
        Gets the value for type.

        Returns:
            int: The current value.
        """
        ...

class Smptebars(FilterBase):
    """
    Generate SMPTE color bars.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...
    def setSar(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for sar.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSar(self) -> Tuple[int, int]:
        """
        Gets the value for sar.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...

class Smptehdbars(FilterBase):
    """
    Generate SMPTE HD color bars.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...
    def setSar(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for sar.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSar(self) -> Tuple[int, int]:
        """
        Gets the value for sar.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...

class Testsrc(FilterBase):
    """
    Generate test pattern.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...
    def setSar(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for sar.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSar(self) -> Tuple[int, int]:
        """
        Gets the value for sar.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setDecimals(self, value: int) -> None:
        """
        Sets the value for decimals.

        Args:
            value (int): The value to set.
        """
        ...
    def getDecimals(self) -> int:
        """
        Gets the value for decimals.

        Returns:
            int: The current value.
        """
        ...

class Testsrc2(FilterBase):
    """
    Generate another test pattern.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...
    def setSar(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for sar.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSar(self) -> Tuple[int, int]:
        """
        Gets the value for sar.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setAlpha(self, value: int) -> None:
        """
        Sets the value for alpha.

        Args:
            value (int): The value to set.
        """
        ...
    def getAlpha(self) -> int:
        """
        Gets the value for alpha.

        Returns:
            int: The current value.
        """
        ...

class Yuvtestsrc(FilterBase):
    """
    Generate YUV test pattern.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...
    def setSar(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for sar.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSar(self) -> Tuple[int, int]:
        """
        Gets the value for sar.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...

class Zoneplate(FilterBase):
    """
    Generate zone-plate.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...
    def setSar(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for sar.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSar(self) -> Tuple[int, int]:
        """
        Gets the value for sar.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setPrecision(self, value: int) -> None:
        """
        Sets the value for precision.

        Args:
            value (int): The value to set.
        """
        ...
    def getPrecision(self) -> int:
        """
        Gets the value for precision.

        Returns:
            int: The current value.
        """
        ...
    def setXo(self, value: int) -> None:
        """
        Sets the value for xo.

        Args:
            value (int): The value to set.
        """
        ...
    def getXo(self) -> int:
        """
        Gets the value for xo.

        Returns:
            int: The current value.
        """
        ...
    def setYo(self, value: int) -> None:
        """
        Sets the value for yo.

        Args:
            value (int): The value to set.
        """
        ...
    def getYo(self) -> int:
        """
        Gets the value for yo.

        Returns:
            int: The current value.
        """
        ...
    def setTo(self, value: int) -> None:
        """
        Sets the value for to.

        Args:
            value (int): The value to set.
        """
        ...
    def getTo(self) -> int:
        """
        Gets the value for to.

        Returns:
            int: The current value.
        """
        ...
    def setK0(self, value: int) -> None:
        """
        Sets the value for k0.

        Args:
            value (int): The value to set.
        """
        ...
    def getK0(self) -> int:
        """
        Gets the value for k0.

        Returns:
            int: The current value.
        """
        ...
    def setKx(self, value: int) -> None:
        """
        Sets the value for kx.

        Args:
            value (int): The value to set.
        """
        ...
    def getKx(self) -> int:
        """
        Gets the value for kx.

        Returns:
            int: The current value.
        """
        ...
    def setKy(self, value: int) -> None:
        """
        Sets the value for ky.

        Args:
            value (int): The value to set.
        """
        ...
    def getKy(self) -> int:
        """
        Gets the value for ky.

        Returns:
            int: The current value.
        """
        ...
    def setKt(self, value: int) -> None:
        """
        Sets the value for kt.

        Args:
            value (int): The value to set.
        """
        ...
    def getKt(self) -> int:
        """
        Gets the value for kt.

        Returns:
            int: The current value.
        """
        ...
    def setKxt(self, value: int) -> None:
        """
        Sets the value for kxt.

        Args:
            value (int): The value to set.
        """
        ...
    def getKxt(self) -> int:
        """
        Gets the value for kxt.

        Returns:
            int: The current value.
        """
        ...
    def setKyt(self, value: int) -> None:
        """
        Sets the value for kyt.

        Args:
            value (int): The value to set.
        """
        ...
    def getKyt(self) -> int:
        """
        Gets the value for kyt.

        Returns:
            int: The current value.
        """
        ...
    def setKxy(self, value: int) -> None:
        """
        Sets the value for kxy.

        Args:
            value (int): The value to set.
        """
        ...
    def getKxy(self) -> int:
        """
        Gets the value for kxy.

        Returns:
            int: The current value.
        """
        ...
    def setKx2(self, value: int) -> None:
        """
        Sets the value for kx2.

        Args:
            value (int): The value to set.
        """
        ...
    def getKx2(self) -> int:
        """
        Gets the value for kx2.

        Returns:
            int: The current value.
        """
        ...
    def setKy2(self, value: int) -> None:
        """
        Sets the value for ky2.

        Args:
            value (int): The value to set.
        """
        ...
    def getKy2(self) -> int:
        """
        Gets the value for ky2.

        Returns:
            int: The current value.
        """
        ...
    def setKt2(self, value: int) -> None:
        """
        Sets the value for kt2.

        Args:
            value (int): The value to set.
        """
        ...
    def getKt2(self) -> int:
        """
        Gets the value for kt2.

        Returns:
            int: The current value.
        """
        ...
    def setKu(self, value: int) -> None:
        """
        Sets the value for ku.

        Args:
            value (int): The value to set.
        """
        ...
    def getKu(self) -> int:
        """
        Gets the value for ku.

        Returns:
            int: The current value.
        """
        ...
    def setKv(self, value: int) -> None:
        """
        Sets the value for kv.

        Args:
            value (int): The value to set.
        """
        ...
    def getKv(self) -> int:
        """
        Gets the value for kv.

        Returns:
            int: The current value.
        """
        ...

class Nullsink(FilterBase):
    """
    Do absolutely nothing with the input video.
    """
    pass

class A3dscope(FilterBase):
    """
    Convert input audio to 3d scope video output.
    """
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setFov(self, value: float) -> None:
        """
        Sets the value for fov.

        Args:
            value (float): The value to set.
        """
        ...
    def getFov(self) -> float:
        """
        Gets the value for fov.

        Returns:
            float: The current value.
        """
        ...
    def setRoll(self, value: float) -> None:
        """
        Sets the value for roll.

        Args:
            value (float): The value to set.
        """
        ...
    def getRoll(self) -> float:
        """
        Gets the value for roll.

        Returns:
            float: The current value.
        """
        ...
    def setPitch(self, value: float) -> None:
        """
        Sets the value for pitch.

        Args:
            value (float): The value to set.
        """
        ...
    def getPitch(self) -> float:
        """
        Gets the value for pitch.

        Returns:
            float: The current value.
        """
        ...
    def setYaw(self, value: float) -> None:
        """
        Sets the value for yaw.

        Args:
            value (float): The value to set.
        """
        ...
    def getYaw(self) -> float:
        """
        Gets the value for yaw.

        Returns:
            float: The current value.
        """
        ...
    def setZzoom(self, value: float) -> None:
        """
        Sets the value for zzoom.

        Args:
            value (float): The value to set.
        """
        ...
    def getZzoom(self) -> float:
        """
        Gets the value for zzoom.

        Returns:
            float: The current value.
        """
        ...
    def setZpos(self, value: float) -> None:
        """
        Sets the value for zpos.

        Args:
            value (float): The value to set.
        """
        ...
    def getZpos(self) -> float:
        """
        Gets the value for zpos.

        Returns:
            float: The current value.
        """
        ...
    def setLength(self, value: int) -> None:
        """
        Sets the value for length.

        Args:
            value (int): The value to set.
        """
        ...
    def getLength(self) -> int:
        """
        Gets the value for length.

        Returns:
            int: The current value.
        """
        ...

class Abitscope(FilterBase):
    """
    Convert input audio to audio bit scope video output.
    """
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setColors(self, value: Any) -> None:
        """
        Sets the value for colors.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColors(self) -> Any:
        """
        Gets the value for colors.

        Returns:
            Any: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...

class Adrawgraph(FilterBase):
    """
    Draw a graph using input audio metadata.
    """
    def setM1(self, value: Any) -> None:
        """
        Sets the value for m1.

        Args:
            value (Any): The value to set.
        """
        ...
    def getM1(self) -> Any:
        """
        Gets the value for m1.

        Returns:
            Any: The current value.
        """
        ...
    def setFg1(self, value: Any) -> None:
        """
        Sets the value for fg1.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFg1(self) -> Any:
        """
        Gets the value for fg1.

        Returns:
            Any: The current value.
        """
        ...
    def setM2(self, value: Any) -> None:
        """
        Sets the value for m2.

        Args:
            value (Any): The value to set.
        """
        ...
    def getM2(self) -> Any:
        """
        Gets the value for m2.

        Returns:
            Any: The current value.
        """
        ...
    def setFg2(self, value: Any) -> None:
        """
        Sets the value for fg2.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFg2(self) -> Any:
        """
        Gets the value for fg2.

        Returns:
            Any: The current value.
        """
        ...
    def setM3(self, value: Any) -> None:
        """
        Sets the value for m3.

        Args:
            value (Any): The value to set.
        """
        ...
    def getM3(self) -> Any:
        """
        Gets the value for m3.

        Returns:
            Any: The current value.
        """
        ...
    def setFg3(self, value: Any) -> None:
        """
        Sets the value for fg3.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFg3(self) -> Any:
        """
        Gets the value for fg3.

        Returns:
            Any: The current value.
        """
        ...
    def setM4(self, value: Any) -> None:
        """
        Sets the value for m4.

        Args:
            value (Any): The value to set.
        """
        ...
    def getM4(self) -> Any:
        """
        Gets the value for m4.

        Returns:
            Any: The current value.
        """
        ...
    def setFg4(self, value: Any) -> None:
        """
        Sets the value for fg4.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFg4(self) -> Any:
        """
        Gets the value for fg4.

        Returns:
            Any: The current value.
        """
        ...
    def setBg(self, value: Any) -> None:
        """
        Sets the value for bg.

        Args:
            value (Any): The value to set.
        """
        ...
    def getBg(self) -> Any:
        """
        Gets the value for bg.

        Returns:
            Any: The current value.
        """
        ...
    def setMin(self, value: float) -> None:
        """
        Sets the value for min.

        Args:
            value (float): The value to set.
        """
        ...
    def getMin(self) -> float:
        """
        Gets the value for min.

        Returns:
            float: The current value.
        """
        ...
    def setMax(self, value: float) -> None:
        """
        Sets the value for max.

        Args:
            value (float): The value to set.
        """
        ...
    def getMax(self) -> float:
        """
        Gets the value for max.

        Returns:
            float: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setSlide(self, value: int) -> None:
        """
        Sets the value for slide.

        Args:
            value (int): The value to set.
        """
        ...
    def getSlide(self) -> int:
        """
        Gets the value for slide.

        Returns:
            int: The current value.
        """
        ...
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...

class Agraphmonitor(FilterBase):
    """
    Show various filtergraph stats.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setOpacity(self, value: float) -> None:
        """
        Sets the value for opacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getOpacity(self) -> float:
        """
        Gets the value for opacity.

        Returns:
            float: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setFlags(self, value: int) -> None:
        """
        Sets the value for flags.

        Args:
            value (int): The value to set.
        """
        ...
    def getFlags(self) -> int:
        """
        Gets the value for flags.

        Returns:
            int: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...

class Ahistogram(FilterBase):
    """
    Convert input audio to histogram video output.
    """
    def setDmode(self, value: int) -> None:
        """
        Sets the value for dmode.

        Args:
            value (int): The value to set.
        """
        ...
    def getDmode(self) -> int:
        """
        Gets the value for dmode.

        Returns:
            int: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setScale(self, value: int) -> None:
        """
        Sets the value for scale.

        Args:
            value (int): The value to set.
        """
        ...
    def getScale(self) -> int:
        """
        Gets the value for scale.

        Returns:
            int: The current value.
        """
        ...
    def setAscale(self, value: int) -> None:
        """
        Sets the value for ascale.

        Args:
            value (int): The value to set.
        """
        ...
    def getAscale(self) -> int:
        """
        Gets the value for ascale.

        Returns:
            int: The current value.
        """
        ...
    def setAcount(self, value: int) -> None:
        """
        Sets the value for acount.

        Args:
            value (int): The value to set.
        """
        ...
    def getAcount(self) -> int:
        """
        Gets the value for acount.

        Returns:
            int: The current value.
        """
        ...
    def setRheight(self, value: float) -> None:
        """
        Sets the value for rheight.

        Args:
            value (float): The value to set.
        """
        ...
    def getRheight(self) -> float:
        """
        Gets the value for rheight.

        Returns:
            float: The current value.
        """
        ...
    def setSlide(self, value: int) -> None:
        """
        Sets the value for slide.

        Args:
            value (int): The value to set.
        """
        ...
    def getSlide(self) -> int:
        """
        Gets the value for slide.

        Returns:
            int: The current value.
        """
        ...
    def setHmode(self, value: int) -> None:
        """
        Sets the value for hmode.

        Args:
            value (int): The value to set.
        """
        ...
    def getHmode(self) -> int:
        """
        Gets the value for hmode.

        Returns:
            int: The current value.
        """
        ...

class Aphasemeter(FilterBase):
    """
    Convert input audio to phase meter video output.
    """
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRc(self, value: int) -> None:
        """
        Sets the value for rc.

        Args:
            value (int): The value to set.
        """
        ...
    def getRc(self) -> int:
        """
        Gets the value for rc.

        Returns:
            int: The current value.
        """
        ...
    def setGc(self, value: int) -> None:
        """
        Sets the value for gc.

        Args:
            value (int): The value to set.
        """
        ...
    def getGc(self) -> int:
        """
        Gets the value for gc.

        Returns:
            int: The current value.
        """
        ...
    def setBc(self, value: int) -> None:
        """
        Sets the value for bc.

        Args:
            value (int): The value to set.
        """
        ...
    def getBc(self) -> int:
        """
        Gets the value for bc.

        Returns:
            int: The current value.
        """
        ...
    def setMpc(self, value: Any) -> None:
        """
        Sets the value for mpc.

        Args:
            value (Any): The value to set.
        """
        ...
    def getMpc(self) -> Any:
        """
        Gets the value for mpc.

        Returns:
            Any: The current value.
        """
        ...
    def setVideo(self, value: bool) -> None:
        """
        Sets the value for video.

        Args:
            value (bool): The value to set.
        """
        ...
    def getVideo(self) -> bool:
        """
        Gets the value for video.

        Returns:
            bool: The current value.
        """
        ...
    def setPhasing(self, value: bool) -> None:
        """
        Sets the value for phasing.

        Args:
            value (bool): The value to set.
        """
        ...
    def getPhasing(self) -> bool:
        """
        Gets the value for phasing.

        Returns:
            bool: The current value.
        """
        ...
    def setTolerance(self, value: float) -> None:
        """
        Sets the value for tolerance.

        Args:
            value (float): The value to set.
        """
        ...
    def getTolerance(self) -> float:
        """
        Gets the value for tolerance.

        Returns:
            float: The current value.
        """
        ...
    def setAngle(self, value: float) -> None:
        """
        Sets the value for angle.

        Args:
            value (float): The value to set.
        """
        ...
    def getAngle(self) -> float:
        """
        Gets the value for angle.

        Returns:
            float: The current value.
        """
        ...
    def setDuration(self, value: int) -> None:
        """
        Sets the value for duration.

        Args:
            value (int): The value to set.
        """
        ...
    def getDuration(self) -> int:
        """
        Gets the value for duration.

        Returns:
            int: The current value.
        """
        ...

class Avectorscope(FilterBase):
    """
    Convert input audio to vectorscope video output.
    """
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRc(self, value: int) -> None:
        """
        Sets the value for rc.

        Args:
            value (int): The value to set.
        """
        ...
    def getRc(self) -> int:
        """
        Gets the value for rc.

        Returns:
            int: The current value.
        """
        ...
    def setGc(self, value: int) -> None:
        """
        Sets the value for gc.

        Args:
            value (int): The value to set.
        """
        ...
    def getGc(self) -> int:
        """
        Gets the value for gc.

        Returns:
            int: The current value.
        """
        ...
    def setBc(self, value: int) -> None:
        """
        Sets the value for bc.

        Args:
            value (int): The value to set.
        """
        ...
    def getBc(self) -> int:
        """
        Gets the value for bc.

        Returns:
            int: The current value.
        """
        ...
    def setAc(self, value: int) -> None:
        """
        Sets the value for ac.

        Args:
            value (int): The value to set.
        """
        ...
    def getAc(self) -> int:
        """
        Gets the value for ac.

        Returns:
            int: The current value.
        """
        ...
    def setRf(self, value: int) -> None:
        """
        Sets the value for rf.

        Args:
            value (int): The value to set.
        """
        ...
    def getRf(self) -> int:
        """
        Gets the value for rf.

        Returns:
            int: The current value.
        """
        ...
    def setGf(self, value: int) -> None:
        """
        Sets the value for gf.

        Args:
            value (int): The value to set.
        """
        ...
    def getGf(self) -> int:
        """
        Gets the value for gf.

        Returns:
            int: The current value.
        """
        ...
    def setBf(self, value: int) -> None:
        """
        Sets the value for bf.

        Args:
            value (int): The value to set.
        """
        ...
    def getBf(self) -> int:
        """
        Gets the value for bf.

        Returns:
            int: The current value.
        """
        ...
    def setAf(self, value: int) -> None:
        """
        Sets the value for af.

        Args:
            value (int): The value to set.
        """
        ...
    def getAf(self) -> int:
        """
        Gets the value for af.

        Returns:
            int: The current value.
        """
        ...
    def setZoom(self, value: float) -> None:
        """
        Sets the value for zoom.

        Args:
            value (float): The value to set.
        """
        ...
    def getZoom(self) -> float:
        """
        Gets the value for zoom.

        Returns:
            float: The current value.
        """
        ...
    def setDraw(self, value: int) -> None:
        """
        Sets the value for draw.

        Args:
            value (int): The value to set.
        """
        ...
    def getDraw(self) -> int:
        """
        Gets the value for draw.

        Returns:
            int: The current value.
        """
        ...
    def setScale(self, value: int) -> None:
        """
        Sets the value for scale.

        Args:
            value (int): The value to set.
        """
        ...
    def getScale(self) -> int:
        """
        Gets the value for scale.

        Returns:
            int: The current value.
        """
        ...
    def setSwap(self, value: bool) -> None:
        """
        Sets the value for swap.

        Args:
            value (bool): The value to set.
        """
        ...
    def getSwap(self) -> bool:
        """
        Gets the value for swap.

        Returns:
            bool: The current value.
        """
        ...
    def setMirror(self, value: int) -> None:
        """
        Sets the value for mirror.

        Args:
            value (int): The value to set.
        """
        ...
    def getMirror(self) -> int:
        """
        Gets the value for mirror.

        Returns:
            int: The current value.
        """
        ...

class Showcqt(FilterBase):
    """
    Convert input audio to a CQT (Constant/Clamped Q Transform) spectrum video output.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setBar_h(self, value: int) -> None:
        """
        Sets the value for bar_h.

        Args:
            value (int): The value to set.
        """
        ...
    def getBar_h(self) -> int:
        """
        Gets the value for bar_h.

        Returns:
            int: The current value.
        """
        ...
    def setAxis_h(self, value: int) -> None:
        """
        Sets the value for axis_h.

        Args:
            value (int): The value to set.
        """
        ...
    def getAxis_h(self) -> int:
        """
        Gets the value for axis_h.

        Returns:
            int: The current value.
        """
        ...
    def setSono_h(self, value: int) -> None:
        """
        Sets the value for sono_h.

        Args:
            value (int): The value to set.
        """
        ...
    def getSono_h(self) -> int:
        """
        Gets the value for sono_h.

        Returns:
            int: The current value.
        """
        ...
    def setFullhd(self, value: bool) -> None:
        """
        Sets the value for fullhd.

        Args:
            value (bool): The value to set.
        """
        ...
    def getFullhd(self) -> bool:
        """
        Gets the value for fullhd.

        Returns:
            bool: The current value.
        """
        ...
    def setVolume(self, value: Any) -> None:
        """
        Sets the value for volume.

        Args:
            value (Any): The value to set.
        """
        ...
    def getVolume(self) -> Any:
        """
        Gets the value for volume.

        Returns:
            Any: The current value.
        """
        ...
    def setVolume2(self, value: Any) -> None:
        """
        Sets the value for volume2.

        Args:
            value (Any): The value to set.
        """
        ...
    def getVolume2(self) -> Any:
        """
        Gets the value for volume2.

        Returns:
            Any: The current value.
        """
        ...
    def setSono_g(self, value: float) -> None:
        """
        Sets the value for sono_g.

        Args:
            value (float): The value to set.
        """
        ...
    def getSono_g(self) -> float:
        """
        Gets the value for sono_g.

        Returns:
            float: The current value.
        """
        ...
    def setGamma2(self, value: float) -> None:
        """
        Sets the value for gamma2.

        Args:
            value (float): The value to set.
        """
        ...
    def getGamma2(self) -> float:
        """
        Gets the value for gamma2.

        Returns:
            float: The current value.
        """
        ...
    def setBar_t(self, value: float) -> None:
        """
        Sets the value for bar_t.

        Args:
            value (float): The value to set.
        """
        ...
    def getBar_t(self) -> float:
        """
        Gets the value for bar_t.

        Returns:
            float: The current value.
        """
        ...
    def setTimeclamp(self, value: float) -> None:
        """
        Sets the value for timeclamp.

        Args:
            value (float): The value to set.
        """
        ...
    def getTimeclamp(self) -> float:
        """
        Gets the value for timeclamp.

        Returns:
            float: The current value.
        """
        ...
    def setAttack(self, value: float) -> None:
        """
        Sets the value for attack.

        Args:
            value (float): The value to set.
        """
        ...
    def getAttack(self) -> float:
        """
        Gets the value for attack.

        Returns:
            float: The current value.
        """
        ...
    def setBasefreq(self, value: float) -> None:
        """
        Sets the value for basefreq.

        Args:
            value (float): The value to set.
        """
        ...
    def getBasefreq(self) -> float:
        """
        Gets the value for basefreq.

        Returns:
            float: The current value.
        """
        ...
    def setEndfreq(self, value: float) -> None:
        """
        Sets the value for endfreq.

        Args:
            value (float): The value to set.
        """
        ...
    def getEndfreq(self) -> float:
        """
        Gets the value for endfreq.

        Returns:
            float: The current value.
        """
        ...
    def setCoeffclamp(self, value: float) -> None:
        """
        Sets the value for coeffclamp.

        Args:
            value (float): The value to set.
        """
        ...
    def getCoeffclamp(self) -> float:
        """
        Gets the value for coeffclamp.

        Returns:
            float: The current value.
        """
        ...
    def setTlength(self, value: Any) -> None:
        """
        Sets the value for tlength.

        Args:
            value (Any): The value to set.
        """
        ...
    def getTlength(self) -> Any:
        """
        Gets the value for tlength.

        Returns:
            Any: The current value.
        """
        ...
    def setCount(self, value: int) -> None:
        """
        Sets the value for count.

        Args:
            value (int): The value to set.
        """
        ...
    def getCount(self) -> int:
        """
        Gets the value for count.

        Returns:
            int: The current value.
        """
        ...
    def setFcount(self, value: int) -> None:
        """
        Sets the value for fcount.

        Args:
            value (int): The value to set.
        """
        ...
    def getFcount(self) -> int:
        """
        Gets the value for fcount.

        Returns:
            int: The current value.
        """
        ...
    def setFontfile(self, value: Any) -> None:
        """
        Sets the value for fontfile.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFontfile(self) -> Any:
        """
        Gets the value for fontfile.

        Returns:
            Any: The current value.
        """
        ...
    def setFont(self, value: Any) -> None:
        """
        Sets the value for font.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFont(self) -> Any:
        """
        Gets the value for font.

        Returns:
            Any: The current value.
        """
        ...
    def setFontcolor(self, value: Any) -> None:
        """
        Sets the value for fontcolor.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFontcolor(self) -> Any:
        """
        Gets the value for fontcolor.

        Returns:
            Any: The current value.
        """
        ...
    def setAxisfile(self, value: Any) -> None:
        """
        Sets the value for axisfile.

        Args:
            value (Any): The value to set.
        """
        ...
    def getAxisfile(self) -> Any:
        """
        Gets the value for axisfile.

        Returns:
            Any: The current value.
        """
        ...
    def setText(self, value: bool) -> None:
        """
        Sets the value for text.

        Args:
            value (bool): The value to set.
        """
        ...
    def getText(self) -> bool:
        """
        Gets the value for text.

        Returns:
            bool: The current value.
        """
        ...
    def setCsp(self, value: int) -> None:
        """
        Sets the value for csp.

        Args:
            value (int): The value to set.
        """
        ...
    def getCsp(self) -> int:
        """
        Gets the value for csp.

        Returns:
            int: The current value.
        """
        ...
    def setCscheme(self, value: Any) -> None:
        """
        Sets the value for cscheme.

        Args:
            value (Any): The value to set.
        """
        ...
    def getCscheme(self) -> Any:
        """
        Gets the value for cscheme.

        Returns:
            Any: The current value.
        """
        ...

class Showcwt(FilterBase):
    """
    Convert input audio to a CWT (Continuous Wavelet Transform) spectrum video output.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Any) -> None:
        """
        Sets the value for rate.

        Args:
            value (Any): The value to set.
        """
        ...
    def getRate(self) -> Any:
        """
        Gets the value for rate.

        Returns:
            Any: The current value.
        """
        ...
    def setScale(self, value: int) -> None:
        """
        Sets the value for scale.

        Args:
            value (int): The value to set.
        """
        ...
    def getScale(self) -> int:
        """
        Gets the value for scale.

        Returns:
            int: The current value.
        """
        ...
    def setIscale(self, value: int) -> None:
        """
        Sets the value for iscale.

        Args:
            value (int): The value to set.
        """
        ...
    def getIscale(self) -> int:
        """
        Gets the value for iscale.

        Returns:
            int: The current value.
        """
        ...
    def setMin(self, value: float) -> None:
        """
        Sets the value for min.

        Args:
            value (float): The value to set.
        """
        ...
    def getMin(self) -> float:
        """
        Gets the value for min.

        Returns:
            float: The current value.
        """
        ...
    def setMax(self, value: float) -> None:
        """
        Sets the value for max.

        Args:
            value (float): The value to set.
        """
        ...
    def getMax(self) -> float:
        """
        Gets the value for max.

        Returns:
            float: The current value.
        """
        ...
    def setImin(self, value: float) -> None:
        """
        Sets the value for imin.

        Args:
            value (float): The value to set.
        """
        ...
    def getImin(self) -> float:
        """
        Gets the value for imin.

        Returns:
            float: The current value.
        """
        ...
    def setImax(self, value: float) -> None:
        """
        Sets the value for imax.

        Args:
            value (float): The value to set.
        """
        ...
    def getImax(self) -> float:
        """
        Gets the value for imax.

        Returns:
            float: The current value.
        """
        ...
    def setLogb(self, value: float) -> None:
        """
        Sets the value for logb.

        Args:
            value (float): The value to set.
        """
        ...
    def getLogb(self) -> float:
        """
        Gets the value for logb.

        Returns:
            float: The current value.
        """
        ...
    def setDeviation(self, value: float) -> None:
        """
        Sets the value for deviation.

        Args:
            value (float): The value to set.
        """
        ...
    def getDeviation(self) -> float:
        """
        Gets the value for deviation.

        Returns:
            float: The current value.
        """
        ...
    def setPps(self, value: int) -> None:
        """
        Sets the value for pps.

        Args:
            value (int): The value to set.
        """
        ...
    def getPps(self) -> int:
        """
        Gets the value for pps.

        Returns:
            int: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setSlide(self, value: int) -> None:
        """
        Sets the value for slide.

        Args:
            value (int): The value to set.
        """
        ...
    def getSlide(self) -> int:
        """
        Gets the value for slide.

        Returns:
            int: The current value.
        """
        ...
    def setDirection(self, value: int) -> None:
        """
        Sets the value for direction.

        Args:
            value (int): The value to set.
        """
        ...
    def getDirection(self) -> int:
        """
        Gets the value for direction.

        Returns:
            int: The current value.
        """
        ...
    def setBar(self, value: float) -> None:
        """
        Sets the value for bar.

        Args:
            value (float): The value to set.
        """
        ...
    def getBar(self) -> float:
        """
        Gets the value for bar.

        Returns:
            float: The current value.
        """
        ...
    def setRotation(self, value: float) -> None:
        """
        Sets the value for rotation.

        Args:
            value (float): The value to set.
        """
        ...
    def getRotation(self) -> float:
        """
        Gets the value for rotation.

        Returns:
            float: The current value.
        """
        ...

class Showfreqs(FilterBase):
    """
    Convert input audio to a frequencies video output.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setAscale(self, value: int) -> None:
        """
        Sets the value for ascale.

        Args:
            value (int): The value to set.
        """
        ...
    def getAscale(self) -> int:
        """
        Gets the value for ascale.

        Returns:
            int: The current value.
        """
        ...
    def setFscale(self, value: int) -> None:
        """
        Sets the value for fscale.

        Args:
            value (int): The value to set.
        """
        ...
    def getFscale(self) -> int:
        """
        Gets the value for fscale.

        Returns:
            int: The current value.
        """
        ...
    def setWin_size(self, value: int) -> None:
        """
        Sets the value for win_size.

        Args:
            value (int): The value to set.
        """
        ...
    def getWin_size(self) -> int:
        """
        Gets the value for win_size.

        Returns:
            int: The current value.
        """
        ...
    def setWin_func(self, value: int) -> None:
        """
        Sets the value for win_func.

        Args:
            value (int): The value to set.
        """
        ...
    def getWin_func(self) -> int:
        """
        Gets the value for win_func.

        Returns:
            int: The current value.
        """
        ...
    def setOverlap(self, value: float) -> None:
        """
        Sets the value for overlap.

        Args:
            value (float): The value to set.
        """
        ...
    def getOverlap(self) -> float:
        """
        Gets the value for overlap.

        Returns:
            float: The current value.
        """
        ...
    def setAveraging(self, value: int) -> None:
        """
        Sets the value for averaging.

        Args:
            value (int): The value to set.
        """
        ...
    def getAveraging(self) -> int:
        """
        Gets the value for averaging.

        Returns:
            int: The current value.
        """
        ...
    def setColors(self, value: Any) -> None:
        """
        Sets the value for colors.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColors(self) -> Any:
        """
        Gets the value for colors.

        Returns:
            Any: The current value.
        """
        ...
    def setCmode(self, value: int) -> None:
        """
        Sets the value for cmode.

        Args:
            value (int): The value to set.
        """
        ...
    def getCmode(self) -> int:
        """
        Gets the value for cmode.

        Returns:
            int: The current value.
        """
        ...
    def setMinamp(self, value: float) -> None:
        """
        Sets the value for minamp.

        Args:
            value (float): The value to set.
        """
        ...
    def getMinamp(self) -> float:
        """
        Gets the value for minamp.

        Returns:
            float: The current value.
        """
        ...
    def setData(self, value: int) -> None:
        """
        Sets the value for data.

        Args:
            value (int): The value to set.
        """
        ...
    def getData(self) -> int:
        """
        Gets the value for data.

        Returns:
            int: The current value.
        """
        ...
    def setChannels(self, value: Any) -> None:
        """
        Sets the value for channels.

        Args:
            value (Any): The value to set.
        """
        ...
    def getChannels(self) -> Any:
        """
        Gets the value for channels.

        Returns:
            Any: The current value.
        """
        ...

class Showspatial(FilterBase):
    """
    Convert input audio to a spatial video output.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setWin_size(self, value: int) -> None:
        """
        Sets the value for win_size.

        Args:
            value (int): The value to set.
        """
        ...
    def getWin_size(self) -> int:
        """
        Gets the value for win_size.

        Returns:
            int: The current value.
        """
        ...
    def setWin_func(self, value: int) -> None:
        """
        Sets the value for win_func.

        Args:
            value (int): The value to set.
        """
        ...
    def getWin_func(self) -> int:
        """
        Gets the value for win_func.

        Returns:
            int: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...

class Showspectrum(FilterBase):
    """
    Convert input audio to a spectrum video output.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setSlide(self, value: int) -> None:
        """
        Sets the value for slide.

        Args:
            value (int): The value to set.
        """
        ...
    def getSlide(self) -> int:
        """
        Gets the value for slide.

        Returns:
            int: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setColor(self, value: int) -> None:
        """
        Sets the value for color.

        Args:
            value (int): The value to set.
        """
        ...
    def getColor(self) -> int:
        """
        Gets the value for color.

        Returns:
            int: The current value.
        """
        ...
    def setScale(self, value: int) -> None:
        """
        Sets the value for scale.

        Args:
            value (int): The value to set.
        """
        ...
    def getScale(self) -> int:
        """
        Gets the value for scale.

        Returns:
            int: The current value.
        """
        ...
    def setFscale(self, value: int) -> None:
        """
        Sets the value for fscale.

        Args:
            value (int): The value to set.
        """
        ...
    def getFscale(self) -> int:
        """
        Gets the value for fscale.

        Returns:
            int: The current value.
        """
        ...
    def setSaturation(self, value: float) -> None:
        """
        Sets the value for saturation.

        Args:
            value (float): The value to set.
        """
        ...
    def getSaturation(self) -> float:
        """
        Gets the value for saturation.

        Returns:
            float: The current value.
        """
        ...
    def setWin_func(self, value: int) -> None:
        """
        Sets the value for win_func.

        Args:
            value (int): The value to set.
        """
        ...
    def getWin_func(self) -> int:
        """
        Gets the value for win_func.

        Returns:
            int: The current value.
        """
        ...
    def setOrientation(self, value: int) -> None:
        """
        Sets the value for orientation.

        Args:
            value (int): The value to set.
        """
        ...
    def getOrientation(self) -> int:
        """
        Gets the value for orientation.

        Returns:
            int: The current value.
        """
        ...
    def setOverlap(self, value: float) -> None:
        """
        Sets the value for overlap.

        Args:
            value (float): The value to set.
        """
        ...
    def getOverlap(self) -> float:
        """
        Gets the value for overlap.

        Returns:
            float: The current value.
        """
        ...
    def setGain(self, value: float) -> None:
        """
        Sets the value for gain.

        Args:
            value (float): The value to set.
        """
        ...
    def getGain(self) -> float:
        """
        Gets the value for gain.

        Returns:
            float: The current value.
        """
        ...
    def setData(self, value: int) -> None:
        """
        Sets the value for data.

        Args:
            value (int): The value to set.
        """
        ...
    def getData(self) -> int:
        """
        Gets the value for data.

        Returns:
            int: The current value.
        """
        ...
    def setRotation(self, value: float) -> None:
        """
        Sets the value for rotation.

        Args:
            value (float): The value to set.
        """
        ...
    def getRotation(self) -> float:
        """
        Gets the value for rotation.

        Returns:
            float: The current value.
        """
        ...
    def setStart(self, value: int) -> None:
        """
        Sets the value for start.

        Args:
            value (int): The value to set.
        """
        ...
    def getStart(self) -> int:
        """
        Gets the value for start.

        Returns:
            int: The current value.
        """
        ...
    def setStop(self, value: int) -> None:
        """
        Sets the value for stop.

        Args:
            value (int): The value to set.
        """
        ...
    def getStop(self) -> int:
        """
        Gets the value for stop.

        Returns:
            int: The current value.
        """
        ...
    def setFps(self, value: Any) -> None:
        """
        Sets the value for fps.

        Args:
            value (Any): The value to set.
        """
        ...
    def getFps(self) -> Any:
        """
        Gets the value for fps.

        Returns:
            Any: The current value.
        """
        ...
    def setLegend(self, value: bool) -> None:
        """
        Sets the value for legend.

        Args:
            value (bool): The value to set.
        """
        ...
    def getLegend(self) -> bool:
        """
        Gets the value for legend.

        Returns:
            bool: The current value.
        """
        ...
    def setDrange(self, value: float) -> None:
        """
        Sets the value for drange.

        Args:
            value (float): The value to set.
        """
        ...
    def getDrange(self) -> float:
        """
        Gets the value for drange.

        Returns:
            float: The current value.
        """
        ...
    def setLimit(self, value: float) -> None:
        """
        Sets the value for limit.

        Args:
            value (float): The value to set.
        """
        ...
    def getLimit(self) -> float:
        """
        Gets the value for limit.

        Returns:
            float: The current value.
        """
        ...
    def setOpacity(self, value: float) -> None:
        """
        Sets the value for opacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getOpacity(self) -> float:
        """
        Gets the value for opacity.

        Returns:
            float: The current value.
        """
        ...

class Showspectrumpic(FilterBase):
    """
    Convert input audio to a spectrum video output single picture.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setColor(self, value: int) -> None:
        """
        Sets the value for color.

        Args:
            value (int): The value to set.
        """
        ...
    def getColor(self) -> int:
        """
        Gets the value for color.

        Returns:
            int: The current value.
        """
        ...
    def setScale(self, value: int) -> None:
        """
        Sets the value for scale.

        Args:
            value (int): The value to set.
        """
        ...
    def getScale(self) -> int:
        """
        Gets the value for scale.

        Returns:
            int: The current value.
        """
        ...
    def setFscale(self, value: int) -> None:
        """
        Sets the value for fscale.

        Args:
            value (int): The value to set.
        """
        ...
    def getFscale(self) -> int:
        """
        Gets the value for fscale.

        Returns:
            int: The current value.
        """
        ...
    def setSaturation(self, value: float) -> None:
        """
        Sets the value for saturation.

        Args:
            value (float): The value to set.
        """
        ...
    def getSaturation(self) -> float:
        """
        Gets the value for saturation.

        Returns:
            float: The current value.
        """
        ...
    def setWin_func(self, value: int) -> None:
        """
        Sets the value for win_func.

        Args:
            value (int): The value to set.
        """
        ...
    def getWin_func(self) -> int:
        """
        Gets the value for win_func.

        Returns:
            int: The current value.
        """
        ...
    def setOrientation(self, value: int) -> None:
        """
        Sets the value for orientation.

        Args:
            value (int): The value to set.
        """
        ...
    def getOrientation(self) -> int:
        """
        Gets the value for orientation.

        Returns:
            int: The current value.
        """
        ...
    def setGain(self, value: float) -> None:
        """
        Sets the value for gain.

        Args:
            value (float): The value to set.
        """
        ...
    def getGain(self) -> float:
        """
        Gets the value for gain.

        Returns:
            float: The current value.
        """
        ...
    def setLegend(self, value: bool) -> None:
        """
        Sets the value for legend.

        Args:
            value (bool): The value to set.
        """
        ...
    def getLegend(self) -> bool:
        """
        Gets the value for legend.

        Returns:
            bool: The current value.
        """
        ...
    def setRotation(self, value: float) -> None:
        """
        Sets the value for rotation.

        Args:
            value (float): The value to set.
        """
        ...
    def getRotation(self) -> float:
        """
        Gets the value for rotation.

        Returns:
            float: The current value.
        """
        ...
    def setStart(self, value: int) -> None:
        """
        Sets the value for start.

        Args:
            value (int): The value to set.
        """
        ...
    def getStart(self) -> int:
        """
        Gets the value for start.

        Returns:
            int: The current value.
        """
        ...
    def setStop(self, value: int) -> None:
        """
        Sets the value for stop.

        Args:
            value (int): The value to set.
        """
        ...
    def getStop(self) -> int:
        """
        Gets the value for stop.

        Returns:
            int: The current value.
        """
        ...
    def setDrange(self, value: float) -> None:
        """
        Sets the value for drange.

        Args:
            value (float): The value to set.
        """
        ...
    def getDrange(self) -> float:
        """
        Gets the value for drange.

        Returns:
            float: The current value.
        """
        ...
    def setLimit(self, value: float) -> None:
        """
        Sets the value for limit.

        Args:
            value (float): The value to set.
        """
        ...
    def getLimit(self) -> float:
        """
        Gets the value for limit.

        Returns:
            float: The current value.
        """
        ...
    def setOpacity(self, value: float) -> None:
        """
        Sets the value for opacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getOpacity(self) -> float:
        """
        Gets the value for opacity.

        Returns:
            float: The current value.
        """
        ...

class Showvolume(FilterBase):
    """
    Convert input audio volume to video output.
    """
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setBorderWidth(self, value: int) -> None:
        """
        Sets the value for borderwidth.

        Args:
            value (int): The value to set.
        """
        ...
    def getBorderWidth(self) -> int:
        """
        Gets the value for borderwidth.

        Returns:
            int: The current value.
        """
        ...
    def setChannelWidth(self, value: int) -> None:
        """
        Sets the value for channelwidth.

        Args:
            value (int): The value to set.
        """
        ...
    def getChannelWidth(self) -> int:
        """
        Gets the value for channelwidth.

        Returns:
            int: The current value.
        """
        ...
    def setChannelHeight(self, value: int) -> None:
        """
        Sets the value for channelheight.

        Args:
            value (int): The value to set.
        """
        ...
    def getChannelHeight(self) -> int:
        """
        Gets the value for channelheight.

        Returns:
            int: The current value.
        """
        ...
    def setFade(self, value: float) -> None:
        """
        Sets the value for fade.

        Args:
            value (float): The value to set.
        """
        ...
    def getFade(self) -> float:
        """
        Gets the value for fade.

        Returns:
            float: The current value.
        """
        ...
    def setVolumeColor(self, value: Any) -> None:
        """
        Sets the value for volumecolor.

        Args:
            value (Any): The value to set.
        """
        ...
    def getVolumeColor(self) -> Any:
        """
        Gets the value for volumecolor.

        Returns:
            Any: The current value.
        """
        ...
    def setDisplayChannelNames(self, value: bool) -> None:
        """
        Sets the value for displaychannelnames.

        Args:
            value (bool): The value to set.
        """
        ...
    def getDisplayChannelNames(self) -> bool:
        """
        Gets the value for displaychannelnames.

        Returns:
            bool: The current value.
        """
        ...
    def setDisplayVolume(self, value: bool) -> None:
        """
        Sets the value for displayvolume.

        Args:
            value (bool): The value to set.
        """
        ...
    def getDisplayVolume(self) -> bool:
        """
        Gets the value for displayvolume.

        Returns:
            bool: The current value.
        """
        ...
    def setDm(self, value: float) -> None:
        """
        Sets the value for dm.

        Args:
            value (float): The value to set.
        """
        ...
    def getDm(self) -> float:
        """
        Gets the value for dm.

        Returns:
            float: The current value.
        """
        ...
    def setDmc(self, value: Any) -> None:
        """
        Sets the value for dmc.

        Args:
            value (Any): The value to set.
        """
        ...
    def getDmc(self) -> Any:
        """
        Gets the value for dmc.

        Returns:
            Any: The current value.
        """
        ...
    def setOrientation(self, value: int) -> None:
        """
        Sets the value for orientation.

        Args:
            value (int): The value to set.
        """
        ...
    def getOrientation(self) -> int:
        """
        Gets the value for orientation.

        Returns:
            int: The current value.
        """
        ...
    def setStepSize(self, value: int) -> None:
        """
        Sets the value for stepsize.

        Args:
            value (int): The value to set.
        """
        ...
    def getStepSize(self) -> int:
        """
        Gets the value for stepsize.

        Returns:
            int: The current value.
        """
        ...
    def setBackgroundOpacity(self, value: float) -> None:
        """
        Sets the value for backgroundopacity.

        Args:
            value (float): The value to set.
        """
        ...
    def getBackgroundOpacity(self) -> float:
        """
        Gets the value for backgroundopacity.

        Returns:
            float: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setDs(self, value: int) -> None:
        """
        Sets the value for ds.

        Args:
            value (int): The value to set.
        """
        ...
    def getDs(self) -> int:
        """
        Gets the value for ds.

        Returns:
            int: The current value.
        """
        ...

class Showwaves(FilterBase):
    """
    Convert input audio to a video output.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setMode(self, value: int) -> None:
        """
        Sets the value for mode.

        Args:
            value (int): The value to set.
        """
        ...
    def getMode(self) -> int:
        """
        Gets the value for mode.

        Returns:
            int: The current value.
        """
        ...
    def setHowManySamplesToShowInTheSamePoint(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for howmanysamplestoshowinthesamepoint.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getHowManySamplesToShowInTheSamePoint(self) -> Tuple[int, int]:
        """
        Gets the value for howmanysamplestoshowinthesamepoint.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setRate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getRate(self) -> Tuple[int, int]:
        """
        Gets the value for rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setSplit_channels(self, value: bool) -> None:
        """
        Sets the value for split_channels.

        Args:
            value (bool): The value to set.
        """
        ...
    def getSplit_channels(self) -> bool:
        """
        Gets the value for split_channels.

        Returns:
            bool: The current value.
        """
        ...
    def setColors(self, value: Any) -> None:
        """
        Sets the value for colors.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColors(self) -> Any:
        """
        Gets the value for colors.

        Returns:
            Any: The current value.
        """
        ...
    def setScale(self, value: int) -> None:
        """
        Sets the value for scale.

        Args:
            value (int): The value to set.
        """
        ...
    def getScale(self) -> int:
        """
        Gets the value for scale.

        Returns:
            int: The current value.
        """
        ...
    def setDraw(self, value: int) -> None:
        """
        Sets the value for draw.

        Args:
            value (int): The value to set.
        """
        ...
    def getDraw(self) -> int:
        """
        Gets the value for draw.

        Returns:
            int: The current value.
        """
        ...

class Showwavespic(FilterBase):
    """
    Convert input audio to a video output single picture.
    """
    def setSize(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getSize(self) -> Tuple[int, int]:
        """
        Gets the value for size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setSplit_channels(self, value: bool) -> None:
        """
        Sets the value for split_channels.

        Args:
            value (bool): The value to set.
        """
        ...
    def getSplit_channels(self) -> bool:
        """
        Gets the value for split_channels.

        Returns:
            bool: The current value.
        """
        ...
    def setColors(self, value: Any) -> None:
        """
        Sets the value for colors.

        Args:
            value (Any): The value to set.
        """
        ...
    def getColors(self) -> Any:
        """
        Gets the value for colors.

        Returns:
            Any: The current value.
        """
        ...
    def setScale(self, value: int) -> None:
        """
        Sets the value for scale.

        Args:
            value (int): The value to set.
        """
        ...
    def getScale(self) -> int:
        """
        Gets the value for scale.

        Returns:
            int: The current value.
        """
        ...
    def setDraw(self, value: int) -> None:
        """
        Sets the value for draw.

        Args:
            value (int): The value to set.
        """
        ...
    def getDraw(self) -> int:
        """
        Gets the value for draw.

        Returns:
            int: The current value.
        """
        ...
    def setFilter(self, value: int) -> None:
        """
        Sets the value for filter.

        Args:
            value (int): The value to set.
        """
        ...
    def getFilter(self) -> int:
        """
        Gets the value for filter.

        Returns:
            int: The current value.
        """
        ...

class Buffer(FilterBase):
    """
    Buffer video frames, and make them accessible to the filterchain.
    """
    def setHeight(self, value: int) -> None:
        """
        Sets the value for height.

        Args:
            value (int): The value to set.
        """
        ...
    def getHeight(self) -> int:
        """
        Gets the value for height.

        Returns:
            int: The current value.
        """
        ...
    def setVideo_size(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for video_size.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getVideo_size(self) -> Tuple[int, int]:
        """
        Gets the value for video_size.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setPix_fmt(self, value: Any) -> None:
        """
        Sets the value for pix_fmt.

        Args:
            value (Any): The value to set.
        """
        ...
    def getPix_fmt(self) -> Any:
        """
        Gets the value for pix_fmt.

        Returns:
            Any: The current value.
        """
        ...
    def setPixel_aspect(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for pixel_aspect.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getPixel_aspect(self) -> Tuple[int, int]:
        """
        Gets the value for pixel_aspect.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setFrame_rate(self, value: Tuple[int, int]) -> None:
        """
        Sets the value for frame_rate.

        Args:
            value (Tuple[int, int]): The value to set.
        """
        ...
    def getFrame_rate(self) -> Tuple[int, int]:
        """
        Gets the value for frame_rate.

        Returns:
            Tuple[int, int]: The current value.
        """
        ...
    def setColorspace(self, value: int) -> None:
        """
        Sets the value for colorspace.

        Args:
            value (int): The value to set.
        """
        ...
    def getColorspace(self) -> int:
        """
        Gets the value for colorspace.

        Returns:
            int: The current value.
        """
        ...
    def setRange(self, value: int) -> None:
        """
        Sets the value for range.

        Args:
            value (int): The value to set.
        """
        ...
    def getRange(self) -> int:
        """
        Gets the value for range.

        Returns:
            int: The current value.
        """
        ...

class Buffersink(FilterBase):
    """
    Buffer video frames, and make them available to the end of the filter graph.
    """
    def setPix_fmts(self, value: List[str]) -> None:
        """
        Sets the value for pix_fmts.

        Args:
            value (List[str]): The value to set.
        """
        ...
    def getPix_fmts(self) -> List[str]:
        """
        Gets the value for pix_fmts.

        Returns:
            List[str]: The current value.
        """
        ...
    def setColor_spaces(self, value: List[int]) -> None:
        """
        Sets the value for color_spaces.

        Args:
            value (List[int]): The value to set.
        """
        ...
    def getColor_spaces(self) -> List[int]:
        """
        Gets the value for color_spaces.

        Returns:
            List[int]: The current value.
        """
        ...
    def setColor_ranges(self, value: List[int]) -> None:
        """
        Sets the value for color_ranges.

        Args:
            value (List[int]): The value to set.
        """
        ...
    def getColor_ranges(self) -> List[int]:
        """
        Gets the value for color_ranges.

        Returns:
            List[int]: The current value.
        """
        ...

class FilterType(Enum):
    Acopy = 'Acopy'
    Aderivative = 'Aderivative'
    Aintegral = 'Aintegral'
    Alatency = 'Alatency'
    Amultiply = 'Amultiply'
    Anull = 'Anull'
    Apsnr = 'Apsnr'
    Areverse = 'Areverse'
    Asdr = 'Asdr'
    Ashowinfo = 'Ashowinfo'
    Asisdr = 'Asisdr'
    Earwax = 'Earwax'
    Volumedetect = 'Volumedetect'
    Anullsink = 'Anullsink'
    Addroi = 'Addroi'
    Alphaextract = 'Alphaextract'
    Alphamerge = 'Alphamerge'
    Amplify = 'Amplify'
    Atadenoise = 'Atadenoise'
    Avgblur = 'Avgblur'
    Backgroundkey = 'Backgroundkey'
    Bbox = 'Bbox'
    Bench = 'Bench'
    Bilateral = 'Bilateral'
    Bitplanenoise = 'Bitplanenoise'
    Blackdetect = 'Blackdetect'
    Blend = 'Blend'
    Blockdetect = 'Blockdetect'
    Blurdetect = 'Blurdetect'
    Bm3d = 'Bm3d'
    Bwdif = 'Bwdif'
    Cas = 'Cas'
    Ccrepack = 'Ccrepack'
    Chromahold = 'Chromahold'
    Chromakey = 'Chromakey'
    Chromanr = 'Chromanr'
    Chromashift = 'Chromashift'
    Ciescope = 'Ciescope'
    Codecview = 'Codecview'
    Colorbalance = 'Colorbalance'
    Colorchannelmixer = 'Colorchannelmixer'
    Colorcontrast = 'Colorcontrast'
    Colorcorrect = 'Colorcorrect'
    Colorize = 'Colorize'
    Colorkey = 'Colorkey'
    Colorhold = 'Colorhold'
    Colorlevels = 'Colorlevels'
    Colormap = 'Colormap'
    Colorspace = 'Colorspace'
    Colortemperature = 'Colortemperature'
    Convolution = 'Convolution'
    Convolve = 'Convolve'
    Copy = 'Copy'
    Corr = 'Corr'
    Crop = 'Crop'
    Curves = 'Curves'
    Datascope = 'Datascope'
    Dblur = 'Dblur'
    Dctdnoiz = 'Dctdnoiz'
    Deband = 'Deband'
    Deblock = 'Deblock'
    Decimate = 'Decimate'
    Deconvolve = 'Deconvolve'
    Dedot = 'Dedot'
    Deflate = 'Deflate'
    Deflicker = 'Deflicker'
    Dejudder = 'Dejudder'
    Derain = 'Derain'
    Deshake = 'Deshake'
    Despill = 'Despill'
    Detelecine = 'Detelecine'
    Dilation = 'Dilation'
    Displace = 'Displace'
    Dnn_classify = 'Dnn_classify'
    Dnn_detect = 'Dnn_detect'
    Dnn_processing = 'Dnn_processing'
    Doubleweave = 'Doubleweave'
    Drawbox = 'Drawbox'
    Drawgraph = 'Drawgraph'
    Drawgrid = 'Drawgrid'
    Edgedetect = 'Edgedetect'
    Elbg = 'Elbg'
    Entropy = 'Entropy'
    Epx = 'Epx'
    Erosion = 'Erosion'
    Estdif = 'Estdif'
    Exposure = 'Exposure'
    Extractplanes = 'Extractplanes'
    Fade = 'Fade'
    Feedback = 'Feedback'
    Fftdnoiz = 'Fftdnoiz'
    Fftfilt = 'Fftfilt'
    Field = 'Field'
    Fieldhint = 'Fieldhint'
    Fieldmatch = 'Fieldmatch'
    Fieldorder = 'Fieldorder'
    Fillborders = 'Fillborders'
    Floodfill = 'Floodfill'
    Format = 'Format'
    Fps = 'Fps'
    Framepack = 'Framepack'
    Framerate = 'Framerate'
    Framestep = 'Framestep'
    Freezedetect = 'Freezedetect'
    Freezeframes = 'Freezeframes'
    Fsync = 'Fsync'
    Gblur = 'Gblur'
    Geq = 'Geq'
    Gradfun = 'Gradfun'
    Graphmonitor = 'Graphmonitor'
    Grayworld = 'Grayworld'
    Greyedge = 'Greyedge'
    Guided = 'Guided'
    Haldclut = 'Haldclut'
    Hflip = 'Hflip'
    Histogram = 'Histogram'
    Hqx = 'Hqx'
    Hstack = 'Hstack'
    Hsvhold = 'Hsvhold'
    Hsvkey = 'Hsvkey'
    Hue = 'Hue'
    Huesaturation = 'Huesaturation'
    Hwdownload = 'Hwdownload'
    Hwmap = 'Hwmap'
    Hwupload = 'Hwupload'
    Hwupload_cuda = 'Hwupload_cuda'
    Hysteresis = 'Hysteresis'
    Identity = 'Identity'
    Idet = 'Idet'
    Il = 'Il'
    Inflate = 'Inflate'
    Interleave = 'Interleave'
    Kirsch = 'Kirsch'
    Lagfun = 'Lagfun'
    Latency = 'Latency'
    Lenscorrection = 'Lenscorrection'
    Limitdiff = 'Limitdiff'
    Limiter = 'Limiter'
    Loop = 'Loop'
    Lumakey = 'Lumakey'
    Lut = 'Lut'
    Lut1d = 'Lut1d'
    Lut2 = 'Lut2'
    Lut3d = 'Lut3d'
    Lutrgb = 'Lutrgb'
    Lutyuv = 'Lutyuv'
    Maskedclamp = 'Maskedclamp'
    Maskedmax = 'Maskedmax'
    Maskedmerge = 'Maskedmerge'
    Maskedmin = 'Maskedmin'
    Maskedthreshold = 'Maskedthreshold'
    Maskfun = 'Maskfun'
    Median = 'Median'
    Mergeplanes = 'Mergeplanes'
    Mestimate = 'Mestimate'
    Metadata = 'Metadata'
    Midequalizer = 'Midequalizer'
    Minterpolate = 'Minterpolate'
    Mix = 'Mix'
    Monochrome = 'Monochrome'
    Morpho = 'Morpho'
    Msad = 'Msad'
    Multiply = 'Multiply'
    Negate = 'Negate'
    Nlmeans = 'Nlmeans'
    Noformat = 'Noformat'
    Noise = 'Noise'
    Normalize = 'Normalize'
    Null = 'Null'
    Oscilloscope = 'Oscilloscope'
    Overlay = 'Overlay'
    Pad = 'Pad'
    Palettegen = 'Palettegen'
    Paletteuse = 'Paletteuse'
    Photosensitivity = 'Photosensitivity'
    Pixdesctest = 'Pixdesctest'
    Pixelize = 'Pixelize'
    Pixscope = 'Pixscope'
    Premultiply = 'Premultiply'
    Prewitt = 'Prewitt'
    Pseudocolor = 'Pseudocolor'
    Psnr = 'Psnr'
    Qp = 'Qp'
    Random = 'Random'
    Readeia608 = 'Readeia608'
    Readvitc = 'Readvitc'
    Remap = 'Remap'
    Removegrain = 'Removegrain'
    Removelogo = 'Removelogo'
    Reverse = 'Reverse'
    Rgbashift = 'Rgbashift'
    Roberts = 'Roberts'
    Rotate = 'Rotate'
    Scale = 'Scale'
    Scale2ref = 'Scale2ref'
    Scdet = 'Scdet'
    Scharr = 'Scharr'
    Scroll = 'Scroll'
    Segment = 'Segment'
    Select = 'Select'
    Selectivecolor = 'Selectivecolor'
    Separatefields = 'Separatefields'
    Setdar = 'Setdar'
    Setfield = 'Setfield'
    Setparams = 'Setparams'
    Setpts = 'Setpts'
    Setrange = 'Setrange'
    Setsar = 'Setsar'
    Settb = 'Settb'
    Shear = 'Shear'
    Showinfo = 'Showinfo'
    Showpalette = 'Showpalette'
    Shuffleframes = 'Shuffleframes'
    Shufflepixels = 'Shufflepixels'
    Shuffleplanes = 'Shuffleplanes'
    Sidedata = 'Sidedata'
    Signalstats = 'Signalstats'
    Siti = 'Siti'
    Sobel = 'Sobel'
    Sr = 'Sr'
    Ssim = 'Ssim'
    Ssim360 = 'Ssim360'
    Swaprect = 'Swaprect'
    Swapuv = 'Swapuv'
    Tblend = 'Tblend'
    Telecine = 'Telecine'
    Thistogram = 'Thistogram'
    Threshold = 'Threshold'
    Thumbnail = 'Thumbnail'
    Tile = 'Tile'
    Tiltandshift = 'Tiltandshift'
    Tlut2 = 'Tlut2'
    Tmedian = 'Tmedian'
    Tmidequalizer = 'Tmidequalizer'
    Tmix = 'Tmix'
    Tonemap = 'Tonemap'
    Tpad = 'Tpad'
    Transpose = 'Transpose'
    Trim = 'Trim'
    Unpremultiply = 'Unpremultiply'
    Unsharp = 'Unsharp'
    Untile = 'Untile'
    V360 = 'V360'
    Varblur = 'Varblur'
    Vectorscope = 'Vectorscope'
    Vflip = 'Vflip'
    Vfrdet = 'Vfrdet'
    Vibrance = 'Vibrance'
    Vif = 'Vif'
    Vignette = 'Vignette'
    Vmafmotion = 'Vmafmotion'
    Vstack = 'Vstack'
    W3fdif = 'W3fdif'
    Waveform = 'Waveform'
    Weave = 'Weave'
    Xbr = 'Xbr'
    Xcorrelate = 'Xcorrelate'
    Xfade = 'Xfade'
    Xmedian = 'Xmedian'
    Xstack = 'Xstack'
    Yadif = 'Yadif'
    Yaepblur = 'Yaepblur'
    Zoompan = 'Zoompan'
    Allrgb = 'Allrgb'
    Allyuv = 'Allyuv'
    Cellauto = 'Cellauto'
    Color = 'Color'
    Colorchart = 'Colorchart'
    Colorspectrum = 'Colorspectrum'
    Ddagrab = 'Ddagrab'
    Gradients = 'Gradients'
    Haldclutsrc = 'Haldclutsrc'
    Life = 'Life'
    Mandelbrot = 'Mandelbrot'
    Nullsrc = 'Nullsrc'
    Pal75bars = 'Pal75bars'
    Pal100bars = 'Pal100bars'
    Rgbtestsrc = 'Rgbtestsrc'
    Sierpinski = 'Sierpinski'
    Smptebars = 'Smptebars'
    Smptehdbars = 'Smptehdbars'
    Testsrc = 'Testsrc'
    Testsrc2 = 'Testsrc2'
    Yuvtestsrc = 'Yuvtestsrc'
    Zoneplate = 'Zoneplate'
    Nullsink = 'Nullsink'
    A3dscope = 'A3dscope'
    Abitscope = 'Abitscope'
    Adrawgraph = 'Adrawgraph'
    Agraphmonitor = 'Agraphmonitor'
    Ahistogram = 'Ahistogram'
    Aphasemeter = 'Aphasemeter'
    Avectorscope = 'Avectorscope'
    Showcqt = 'Showcqt'
    Showcwt = 'Showcwt'
    Showfreqs = 'Showfreqs'
    Showspatial = 'Showspatial'
    Showspectrum = 'Showspectrum'
    Showspectrumpic = 'Showspectrumpic'
    Showvolume = 'Showvolume'
    Showwaves = 'Showwaves'
    Showwavespic = 'Showwavespic'
    Buffer = 'Buffer'
    Buffersink = 'Buffersink'

def CreateFilter(type: FilterType) -> Optional[FilterBase]:
    """
    Creates an instance of the specified filter.

    Args:
        type (FilterType): The type of filter to create.

    Returns:
        Optional[FilterBase]: An instance of the specified filter, or None if the type is invalid.
    """
    ...
