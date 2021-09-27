# https://stackoverflow.com/questions/9421108/how-can-i-compile-cuda-code-then-link-it-to-a-c-project
# https://stackoverflow.com/questions/2124753/how-can-i-use-powershell-with-the-visual-studio-command-prompt

# windows - need to set env vars with vcvars64.bat when using cl from command line -- done when starting powershell with my psprofile

INCSAMPLES_H = -I "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.0\common\inc"

CUB = -I "E:\projects\cucuda\cub-1.14.0\cub"

.PHONY: all
all: remove sort_v2

sort:
	nvcc -arch=sm_50 sort.cu -o sort.exe -Wno-deprecated-gpu-targets $(INCSAMPLES_H) -Xptxas -v -lineinfo
	.\sort.exe > output.txt

sort_v2:
	nvcc -arch=sm_50 sort_v2.cu -o sort_v2.exe -Wno-deprecated-gpu-targets $(INCSAMPLES_H) -Xptxas -v -lineinfo
	.\sort_v2.exe > output_v2.txt

#debug binary for mem check
debug:
	del *.exe *.obj *.lib *.exp *.pdb
	nvcc -arch=sm_50 sort.cu -o sort_debug.exe -G -g -O0 -Wno-deprecated-gpu-targets $(INCSAMPLES_H) -Xptxas -v 
	cuda-memcheck ./sort_debug.exe > output_memcheck.txt

debug_v2:
	del *.exe *.obj *.lib *.exp *.pdb
	nvcc -arch=sm_50 sort_v2.cu -o sort_debugv2.exe -G -g -O0 -Wno-deprecated-gpu-targets $(INCSAMPLES_H) -Xptxas -v 
	cuda-memcheck ./sort_debugv2.exe > output_memcheck.txt

py1:
	E:/programs/Anaconda3/python.exe "e:/projects/cucuda/Learning Cuda repo/sort/mergesortsmall.py"

py2:
	E:/programs/Anaconda3/python.exe "e:/projects/cucuda/Learning Cuda repo/sort/mergesortmedium.py" > py2output.txt


thrust:
	nvcc -arch=sm_50 thrustsort1m.cu -o thrustsort1m.exe -Wno-deprecated-gpu-targets $(INCSAMPLES_H) -Xptxas -v
	.\thrustsort1m.exe > thrustsort1m.txt

.PHONY: remove
remove:
	del *.exe *.obj *.lib *.exp *.pdb