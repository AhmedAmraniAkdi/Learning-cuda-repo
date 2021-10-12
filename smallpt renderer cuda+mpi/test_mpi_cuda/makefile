# https://stackoverflow.com/questions/9421108/how-can-i-compile-cuda-code-then-link-it-to-a-c-project
# https://stackoverflow.com/questions/2124753/how-can-i-use-powershell-with-the-visual-studio-command-prompt

# windows - need to set env vars with vcvars64.bat when using cl from command line -- done when starting powershell with my psprofile

MPI_H = /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include\x64"
MPI_LIB = /machine:x64 /dynamicbase "msmpi.lib" /libpath:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" 
FLAGS = /EHsc

INCSAMPLES_H = -I "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.0\common\inc"
LIBSCUDAPATH = /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64" 
LIBSCUDA = cuda.lib cudart.lib

.PHONY: all
all: remove test_cuda_ms_mpi

test_ms_mpi:
	cl test_msmpi.cpp $(FLAGS) $(MPI_H) $(MPI_LIB)
	mpiexec -np 2 .\test_msmpi.exe
	del test_msmpi.obj test_msmpi.exe

test_cuda_ms_mpi: device_query.obj
	cl test_msmpi.cpp $< $(LIBSCUDA) $(MPI_H) $(FLAGS) /link $(MPI_LIB) $(LIBSCUDAPATH)
	mpiexec -np 2 .\test_msmpi.exe > output_mpi_cuda_query.txt

device_query.obj:
	nvcc -c device_query.cu -arch=sm_50 -Wno-deprecated-gpu-targets $(INCSAMPLES_H)


.PHONY: remove
remove:
	del *.exe *.obj *.lib *.exp *.pdb