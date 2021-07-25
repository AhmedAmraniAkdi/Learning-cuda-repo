#include <iostream>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "conv2dseparable_common.h"

// we will round up the size of img to the multiple of blocksize
// that way no need for if and else
int roundup_2power(int number, int multiple){
    return (number + multiple -1) & -multiple;
}

int main(void){

    std::string image_path = cv::samples::findFile("gray_apple.jpeg");
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    //img to float
    cv::Mat img_float;
    img.convertTo(img_float, CV_32FC1, 1.0/255.0);

    cv::Mat img_padded;
    int row_rup = roundup_2power(img.rows, BLOCKDIM);
    int col_rup = roundup_2power(img.cols, BLOCKDIM);
    cv::copyMakeBorder(img_float, 
                        img_padded, 
                        0, 
                        row_rup - img.rows, 
                        0,
                        col_rup - img.cols,
                        cv::BORDER_CONSTANT, 
                        cv::Scalar(0));

    // gaussian_kernel1D = gaussian_kernel1D conv dirac;
    cv::Mat dirac(cv::Size(1, KERNELRADIUS*2 + 1), CV_32FC1, cv::Scalar(0));
    dirac.at<float>(1, KERNELRADIUS) = 1;
    cv::Mat gaussian_kernel;
    cv::GaussianBlur(dirac, gaussian_kernel, cv::Size(1, KERNELRADIUS*2 + 1), 3);
    
    // variables for host
    float* h_input, *h_output, *h_kernel;
    int buf_size = img_padded.rows * img_padded.cols * sizeof(float);
    int kernel_size = (KERNELRADIUS * 2 + 1) * sizeof(float);
    
    h_input = img_padded.ptr<float>(0);
    h_kernel = gaussian_kernel.ptr<float>(0);
    h_output = (float *)malloc(buf_size);

    processing(h_input, h_output, h_kernel, img_padded.cols, img_padded.rows, KERNELRADIUS);

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