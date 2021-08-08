# https://stackoverflow.com/questions/9421108/how-can-i-compile-cuda-code-then-link-it-to-a-c-project
# https://stackoverflow.com/questions/2124753/how-can-i-use-powershell-with-the-visual-studio-command-prompt

# windows - need to set env vars with vcvars64.bat when using cl from command line -- done when starting powershell with my psprofile

INCSAMPLES_H = -I "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.0\common\inc"

.PHONY: all
all: scan compact split 

scan:
	nvcc -arch=sm_50 scan.cu -Wno-deprecated-gpu-targets $(INCSAMPLES_H)

compact:
	nvcc -arch=sm_50 compact.cu -Wno-deprecated-gpu-targets $(INCSAMPLES_H)

split:
	nvcc -arch=sm_50 split.cu -Wno-deprecated-gpu-targets $(INCSAMPLES_H)

.PHONY: remove
remove:
	del *.exe *.obj *.lib *.exp

