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

smallpt_mpi:
	nvcc -arch=sm_50 smallpt_cuda_mpi.cu -o smallpt_cuda_mpi.exe -Wno-deprecated-gpu-targets $(INCSAMPLES_H) $(OTHER_HEADERS) -Xptxas -v -lineinfo -lcurand
	.\smallpt_cuda_mpi.exe

debug:
	del *.exe *.obj *.lib *.exp *.pdb
	nvcc -arch=sm_50 smallpt_cuda.cu -o smallpt_cuda.exe -G -g -O0 -Wno-deprecated-gpu-targets $(INCSAMPLES_H) $(OTHER_HEADERS) -Xptxas -v -lcurand
	cuda-memcheck ./smallpt_cuda.exe > output_memcheck.txt

test_ms_mpi:
	cl test_msmpi.cpp /EHsc /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include\x64" /link /machine:x64 /dynamicbase "msmpi.lib" /libpath:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" 
	mpiexec -np 2 .\test_msmpi.exe
	del test_msmpi.obj test_msmpi.exe

cuda_ms_mpi_smallpt:



.PHONY: remove
remove:
	del *.exe *.obj *.lib *.exp *.pdb