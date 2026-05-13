# CUDA Image Processing Project

This project performs grayscale image processing using CUDA GPU kernels.

## Features
- GPU-accelerated grayscale conversion
- Parallel pixel-level processing using CUDA kernels
- OpenCV integration for image loading and saving
- Efficient memory transfer between host and device
- 
## Technologies
- CUDA (NVIDIA GPU Programming)
- C++
- OpenCV

## Project Overview
The project converts a colored RGB image into grayscale using CUDA. Each pixel is processed in parallel on the GPU, making the operation much faster for large images.

### Processing Steps:
1. Load input image using OpenCV
2. Allocate memory on GPU
3. Copy image data from CPU → GPU
4. Launch CUDA kernel for grayscale conversion
5. Each thread processes one pixel
6. Copy result back to CPU
7. Save output image

## Grayscale Formula

Each pixel is converted using luminance method:

Gray = 0.299 × R + 0.587 × G + 0.114 × B

## Requirements

- NVIDIA CUDA Toolkit
- C++ Compiler (g++)
- OpenCV (preferably OpenCV 4.x)
- NVIDIA GPU with CUDA support

## Build Instructions

## Using CMake (recommended)

```
mkdir build
cd build
cmake ..
make
~~~

## Using nvcc directly

```
nvcc main.cu -o image_processor `pkg-config --cflags --libs opencv4`
```
## Run the Project

```
./image_processor input.jpg output.jpg
```




## Features
- GPU accelerated grayscale conversion
- Parallel image processing using CUDA kernels
