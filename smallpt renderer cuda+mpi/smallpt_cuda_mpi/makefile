# https://stackoverflow.com/questions/9421108/how-can-i-compile-cuda-code-then-link-it-to-a-c-project
# https://stackoverflow.com/questions/2124753/how-can-i-use-powershell-with-the-visual-studio-command-prompt

# windows - need to set env vars with vcvars64.bat when using cl from command line -- done when starting powershell with my psprofile


MPI_H = /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include\x64"
MPI_LIB = /machine:x64 /dynamicbase "msmpi.lib" /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" 
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

smallpt_cuda_mpi: smallpt_cuda_mpi.obj
	cl smallpt_mpi_cuda.c $(MPI_H) $(FLAGS) /link $(MPI_LIB) $< $(LIBSCUDA) $(LIBSCUDAPATH)
	mpiexec -np $(NUM_PROCCESES) .\smallpt_mpi_cuda.exe

smallpt_cuda_mpi.obj:
	nvcc  -c smallpt_cuda_mpi.cu $(INCSAMPLES_H) $(OTHER_HEADERS) $(ADDITIONAL_LIBS) $(NVCC_FLAGS)

profile:
	mpiexec -np $(NUM_PROCCESES) nvprof --output-profile smallptcudampi.%q{PMI_RANK}.nvprof \
	--process-name "rank %q{PMI_RANK}" --context-name "rank %q{PMI_RANK}" .\smallpt_mpi_cuda.exe

time_it:
	Measure-Command {mpiexec -np $(NUM_PROCCESES) .\smallpt_mpi_cuda.exe } > timing_mpi.txt

.PHONY: remove
remove:
	del *.exe *.obj *.lib *.exp *.pdb
