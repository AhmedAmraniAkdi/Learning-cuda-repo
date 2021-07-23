#include <iostream>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "conv2dseparable_common.h"

#define IMROWS 1204
#define IMCOLS 1880
#define IMCHANNELS 1
#define KERNELRADIUS 16 // 16x2 + 1

int main(void){

    std::string image_path = cv::samples::findFile("gray_apple.jpeg");
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    /*imshow("Display window", img);
    int k = cv::waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        imwrite("gray_apple.jpeg", img);
    }*/

    //img to float
    cv::Mat img_float;
    img.convertTo(img_float, CV_32FC1, 1.0/255.0);

    // gaussian_kernel1D = gaussian_kernel1D conv dirac;
    cv::Mat dirac(cv::Size(1, KERNELRADIUS*2 + 1), CV_32FC1, cv::Scalar(0));
    dirac.at<float>(1, KERNELRADIUS) = 1;
    cv::Mat gaussian_kernel;
    cv::GaussianBlur(dirac, gaussian_kernel, cv::Size(1, KERNELRADIUS*2 + 1), 3);
    
    //std::cout << gaussian_kernel;
    
    // variables for host
    float* h_input, *h_output, *h_kernel;
    int buf_size = IMROWS * IMCOLS * sizeof(float);
    int kernel_size = (KERNELRADIUS*2+1) * sizeof(float);
    
    h_input = img_float.ptr<float>(0);
    h_kernel = gaussian_kernel.ptr<float>(0);
    h_output = (float *)malloc(buf_size);

    processing(h_input, h_output, h_kernel);

    /*imshow("Display window", img);
    int k = cv::waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        imwrite("gray_apple.jpeg", img);
    }*/

    // free
    free(h_output);
    return 0;
}