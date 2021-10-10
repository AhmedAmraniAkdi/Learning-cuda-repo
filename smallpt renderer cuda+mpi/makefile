# https://stackoverflow.com/questions/9421108/how-can-i-compile-cuda-code-then-link-it-to-a-c-project
# https://stackoverflow.com/questions/2124753/how-can-i-use-powershell-with-the-visual-studio-command-prompt

# windows - need to set env vars with vcvars64.bat when using cl from command line -- done when starting powershell with my psprofile

INCSAMPLES_H = -I "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.0\common\inc"
OTHER_HEADERS = -I "spheres_rays.cuh" -I "radiance.cuh"

.PHONY: all
all: remove smallpt

smallpt:
	nvcc -arch=sm_50 smallpt_cuda.cu -o smallpt_cuda.exe -Wno-deprecated-gpu-targets $(INCSAMPLES_H) $(OTHER_HEADERS) -Xptxas -v -lineinfo -lcurand
	.\smallpt_cuda.exe

debug:
	del *.exe *.obj *.lib *.exp *.pdb
	nvcc -arch=sm_50 smallpt_cuda.cu -o smallpt_cuda.exe -G -g -O0 -Wno-deprecated-gpu-targets $(INCSAMPLES_H) $(OTHER_HEADERS) -Xptxas -v -lcurand
	cuda-memcheck ./smallpt_cuda.exe > output_memcheck.txt

.PHONY: remove
remove:
	del *.exe *.obj *.lib *.exp *.pdb