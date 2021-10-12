Kevin Baeson's 99 line C++ path tracer http://www.kevinbeason.com/smallpt/

Goal is to use mpi and cuda on the path tracer program.

Cuda version and multigpu cuda version using mpi on 1 gpu , (ms-mpi, would need to install all cuda on a linux machine for openmpi (cuda aware) and multi process service), (a gpu cluster on aws f1 instances for later).

ms-mpi is not cuda aware (https://github.com/microsoft/Microsoft-MPI/issues/9) like open-mpi.

doesn't matter for now, no need to communicate between gpus for this program - difference would be that on cuda aware implementation we would not need to go through the host to pass buffers.

multi process service will allow overlap of kernel executions since gpu ressources are not fully utilized.

[![img.png](https://i.postimg.cc/02Qtzj5h/img.png)](https://postimg.cc/PNgbctwM)

