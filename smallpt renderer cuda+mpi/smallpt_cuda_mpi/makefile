# https://stackoverflow.com/questions/9421108/how-can-i-compile-cuda-code-then-link-it-to-a-c-project
# https://stackoverflow.com/questions/2124753/how-can-i-use-powershell-with-the-visual-studio-command-prompt

# windows - need to set env vars with vcvars64.bat when using cl from command line -- done when starting powershell with my psprofile


MPI_H = /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include\x64"
MPI_LIB = /machine:x64 /dynamicbase "msmpi.lib" /libpath:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" 
FLAGS = /EHsc

INCSAMPLES_H = -I "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.0\common\inc"
OTHER_HEADERS = -I "spheres_rays.cuh" -I "radiance.cuh"
LIBSCUDAPATH = /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64" 
LIBSCUDA = cuda.lib cudart.lib
ADDITIONAL_LIBS = -lcurand
NVCC_FLAGS = -arch=sm_50 -Wno-deprecated-gpu-targets -Xptxas -v -lineinfo

NUM_PROCCESES = 4

.PHONY: all
all: remove smallpt_cuda_mpi

cuda_ms_mpi_smallpt: cuda_ms_mpi_smallpt.obj
	cl smallpt_mpi_cuda.c $< $(LIBSCUDA) $(MPI_H) $(FLAGS) /link $(MPI_LIB) $(LIBSCUDAPATH)
	mpiexec -np $(NUM_PROCCESES) .\smallpt_mpi_cuda.exe > output_smallpt_mpi_cuda.txt

cuda_ms_mpi_smallpt.obj:
	nvcc  -c smallpt_cuda_mpi.cu $(INCSAMPLES_H) $(OTHER_HEADERS) $(ADDITIONAL_LIBS) $(NVCC_FLAGS)

.PHONY: remove
remove:
	del *.exe *.obj *.lib *.exp *.pdb