# https://stackoverflow.com/questions/9421108/how-can-i-compile-cuda-code-then-link-it-to-a-c-project


INCOPENCV = -L "E:\\programs\\cpp_libs\\opencv\\OpenCV-MinGW-Build-OpenCV-4.1.1-x64\\include"
LIBOPENCV = -L "E:\\programs\\cpp_libs\\opencv\\OpenCV-MinGW-Build-OpenCV-4.1.1-x64\\x64\\mingw\\bin" \                
                -llibopencv_calib3d411 \ 
                -llibopencv_core411 \ 
                -llibopencv_dnn411 \ 
                -llibopencv_features2d411 \ 
                -llibopencv_flann411 \ 
                -llibopencv_highgui411 \ 
                -llibopencv_imgcodecs411 \ 
                -llibopencv_imgproc411 \ 
                -llibopencv_ml411 \ 
                -llibopencv_objdetect411 \ 
                -llibopencv_photo411 \ 
                -llibopencv_stitching411 \ 
                -llibopencv_video411 \ 
                -llibopencv_videoio411 \ 

LIBCUDA = -L "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64" -lcuda -lcudart


all: conv2dseparable

conv2dseparable: cudacode.o
    g++ conv2dseparable.cpp -o conv2dseparable $(LIBCUDA) cudacode.o $(INCOPENCV) $(LIBOPENCV)

cudacode.o:
    nvcc -c -arch=sm_50 conv2dseparable.cu
