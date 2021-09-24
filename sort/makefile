# https://stackoverflow.com/questions/9421108/how-can-i-compile-cuda-code-then-link-it-to-a-c-project
# https://stackoverflow.com/questions/2124753/how-can-i-use-powershell-with-the-visual-studio-command-prompt

# windows - need to set env vars with vcvars64.bat when using cl from command line -- done when starting powershell with my psprofile

INCSAMPLES_H = -I "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.0\common\inc"

.PHONY: all
all: remove sort

sort:
	nvcc -arch=sm_50 sort.cu -o sort.exe -Wno-deprecated-gpu-targets $(INCSAMPLES_H) -Xptxas -v -lineinfo

#debug binary for mem check
debug:
	del *.exe *.obj *.lib *.exp *.pdb
	nvcc -arch=sm_50 sort.cu -o sort_debug.exe -G -g -O0 -Wno-deprecated-gpu-targets $(INCSAMPLES_H) -Xptxas -v 
	cuda-memcheck ./sort_debug.exe > output_memcheck.txt

py:
	& E:/programs/Anaconda3/python.exe "e:/projects/cucuda/Learning Cuda repo/sort/mergesortsmall.py"


.PHONY: remove
remove:
	del *.exe *.obj *.lib *.exp *.pdb