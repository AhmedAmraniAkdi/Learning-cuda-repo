# https://stackoverflow.com/questions/9421108/how-can-i-compile-cuda-code-then-link-it-to-a-c-project

# windows - need to set env vars with vcvars64.bat when using cl from command line

#snatched from visual studio
CL_OPT = /EHsc

INCLUDEOPENCV = /I "E:\\programs\\cpp_libs\\opencv\\OpenCV-MinGW-Build-OpenCV-4.1.1-x64\\include"

LIBDIROPENCV = /LIBPATH:"E:\\programs\\cpp_libs\\opencv\\OpenCV-MinGW-Build-OpenCV-4.1.1-x64\\x64\\mingw\\bin" \

LIBSOPENCV = libopencv_calib3d411 \
libopencv_core411 \
libopencv_dnn411 \
libopencv_features2d411 \
libopencv_flann411 \
libopencv_highgui411 \
libopencv_imgcodecs411 \
libopencv_imgproc411 \
libopencv_ml411 \
libopencv_objdetect411 \
libopencv_photo411 \
libopencv_stitching411 \
libopencv_video411 \
libopencv_videoio411 \

LIBDIRCUDA = /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64" 

LIBSCUDA = cuda cudart

INCSAMPLES_H = -I "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.0\common\inc"

all: main
cuda: conv2dseparable.obj

main: conv2dseparable.obj
	cl $(CL_OPT) main.cpp $< $(LIBSCUDA) $(LIBSOPENCV) $(INCLUDECOPENCV) /link $(LIBDIROPENCV) $(LIBDIRCUDA)

conv2dseparable.obj:
	nvcc -c -arch=sm_50 conv2dseparable.cu -Wno-deprecated-gpu-targets $(INCSAMPLES_H)
