import os
import subprocess
import shutil

# Hardcoded version for both CPU and CUDA builds
VERSION = "0.6.0"

def build_package(is_cuda=False):
    """
    Build the Python package, switching between CPU and CUDA versions.
    :param is_cuda: Boolean indicating if the build is for CUDA.
    """
    if is_cuda:
        print(f"Building CUDA version {VERSION}...")
        setup_file = "gpusetup.py"
        source_dir = "celux_cuda"
        temp_build_dir = "build_cuda"
    else:
        print(f"Building CPU version {VERSION}...")
        setup_file = "cpusetup.py"
        source_dir = "celux"
        temp_build_dir = "build_cpu"

    # Clean build directories
    if os.path.exists(temp_build_dir):
        shutil.rmtree(temp_build_dir)
    os.makedirs(temp_build_dir, exist_ok=True)

    # Copy source code to temporary build directory
    shutil.copytree(source_dir, os.path.join(temp_build_dir, source_dir))

    # Copy setup file and other necessary files
    shutil.copy(setup_file, temp_build_dir)
    shutil.copy("README.md", temp_build_dir)
    shutil.copy("./docs/BENCHMARKS.md", temp_build_dir)
    shutil.copy("./docs/CHANGELOG.md", temp_build_dir)
    shutil.copy("./docs/CONTRIBUTING.md", temp_build_dir)
    shutil.copy("./docs/FAQ.md", temp_build_dir)
    shutil.copy("./docs/GETTINGSTARTED.md", temp_build_dir)
    shutil.copy("./docs/INSTALLATION.md", temp_build_dir)
    
    # Change to temporary build directory
    os.chdir(temp_build_dir)

    # Set the version environment variable
    os.environ["CELUX_VERSION"] = VERSION

    # Run the setup file to build the wheel
    subprocess.run(["python", setup_file, "bdist_wheel"], check=True)

    # Move the built wheel to the main dist directory
    os.makedirs("../dist", exist_ok=True)
    for file in os.listdir("dist"):
        shutil.move(os.path.join("dist", file), "../dist/")

    # Change back to the original directory
    os.chdir("..")

    # Clean up the temporary build directory
    shutil.rmtree(temp_build_dir)

def main():
    # Build the CPU version
    build_package(is_cuda=False)

    # Build the CUDA version
   # build_package(is_cuda=True)

if __name__ == "__main__":
    main()
