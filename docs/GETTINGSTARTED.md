
## 📚 Getting Started

### 🎉 Quick Start

Here's a simple example demonstrating how to use **CeLux** to read video frames and process them:

```python
import celux as cx

def process_frame(frame):
    # Implement your frame processing logic here
    pass

with cx.VideoReader(
    "path/to/input/video.mp4",
) as reader:
    for frame in reader:
        # Frame is a PyTorch tensor in HWC format
        process_frame(frame)
```

**Parameters:**

- `device` (str): Device to use. Can be `"cpu"` or `"cuda"`.

### 📜 Detailed Usage

**CeLux** allows you to efficiently decode and process video frames with ease. Below are some common operations:

#### Initialize VideoReader

```python
reader = cx.VideoReader(
    "path/to/video.mp4",
)
```

#### Iterate Through Frames

```python
for frame in reader:
    # Your processing logic
    pass
```

#### Access Video Properties

```python
properties = reader.get_properties()
print(properties)
```