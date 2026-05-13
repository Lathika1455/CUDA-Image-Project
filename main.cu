#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

using namespace cv;
using namespace std;

__global__ void grayscaleKernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int pixelIndex = idx * channels;

    if (idx < width * height)
    {
        unsigned char r = input[pixelIndex];
        unsigned char g = input[pixelIndex + 1];
        unsigned char b = input[pixelIndex + 2];

        unsigned char gray =
            0.299f * r +
            0.587f * g +
            0.114f * b;

        output[pixelIndex] = gray;
        output[pixelIndex + 1] = gray;
        output[pixelIndex + 2] = gray;
    }
}

int main()
{
    Mat image = imread("input_images/sample.jpg");

    if (image.empty())
    {
        cout << "Image not found!" << endl;
        return -1;
    }

    int size =
        image.rows *
        image.cols *
        image.channels();

    unsigned char *d_input, *d_output;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(
        d_input,
        image.data,
        size,
        cudaMemcpyHostToDevice);

    int threads = 256;

    int blocks =
        (image.rows * image.cols + threads - 1)
        / threads;

    grayscaleKernel<<<blocks, threads>>>(
        d_input,
        d_output,
        image.cols,
        image.rows,
        image.channels());

    cudaMemcpy(
        image.data,
        d_output,
        size,
        cudaMemcpyDeviceToHost);

    imwrite(
        "output_images/output.jpg",
        image);

    cudaFree(d_input);
    cudaFree(d_output);

    cout << "Processing completed!" << endl;

    return 0;
}