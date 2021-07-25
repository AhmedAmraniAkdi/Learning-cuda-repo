# https://stackoverflow.com/questions/9421108/how-can-i-compile-cuda-code-then-link-it-to-a-c-project
# https://stackoverflow.com/questions/2124753/how-can-i-use-powershell-with-the-visual-studio-command-prompt

# windows - need to set env vars with vcvars64.bat when using cl from command line -- done when starting powershell with my psprofile

CL_OPT = /EHsc

##

INCLUDEOPENCV = /I "E:\programs\cpp_libs\opencv\build\include"

##

LIBDIROPENCV = /LIBPATH:"E:\programs\cpp_libs\opencv\build\x64\vc15\lib"

LIBDIRCUDA = /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64" 

####

LIBSOPENCV = opencv_world453.lib

LIBSCUDA = cuda.lib cudart.lib

INCSAMPLES_H = -I "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.0\common\inc"

all: main
cuda: conv2dseparable.obj

main: conv2dseparable.obj
	cl $(CL_OPT) main.cpp $< $(LIBSCUDA) $(LIBSOPENCV) $(INCLUDEOPENCV) /link $(LIBDIROPENCV) $(LIBDIRCUDA)

conv2dseparable.obj:
	nvcc -c -arch=sm_50 conv2dseparable.cu -Wno-deprecated-gpu-targets $(INCSAMPLES_H)
