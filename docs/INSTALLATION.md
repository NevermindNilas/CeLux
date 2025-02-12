
## 📦 Installation
 
### 🤖 PIP installation

**CeLux** offers one installation options tailored to your system's capabilities:

1. **CPU-Only Version:** For systems without CUDA-capable GPUs.

### 🖥️ CPU-Only Installation

Install the CPU version of **CeLux** using `pip`:

```bash
pip install celux
```

```python
import celux #as cx
```

## 🛠️ Building from Source

While **CeLux** is easily installable via `pip`, you might want to build it from source for customization or contributing purposes.

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/Trentonom0r3/celux.git
    cd celux
    ```

2. **Install Dependencies:**

    Ensure all prerequisites are installed. You can use `vcpkg` for managing dependencies on Windows.

3. **Configure the Project with CMake:**

    ```bash
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
    ```

    **Windows Users:** If using Vcpkg, include the toolchain file:

    ```bash
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=<path_to_vcpkg>/scripts/buildsystems/vcpkg.cmake
    ```

4. **Build the Project:**

    ```bash
    cmake --build build --config Release
    ```

5. **Install the Package:**

    ```bash
    cmake --install build
    ```

6. **Set Up Environment Variables:**

    Ensure FFmpeg binaries and other dependencies are in your system's `PATH`. On Unix systems, you might need to set `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH`.