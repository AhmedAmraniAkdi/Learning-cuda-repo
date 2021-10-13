Kevin Baeson's 99 line C++ path tracer http://www.kevinbeason.com/smallpt/

Goal is to use mpi and cuda on the path tracer program.

Cuda version and multigpu cuda version using mpi on 1 gpu , (ms-mpi, would need to install all cuda on a linux machine for openmpi (cuda aware) and multi process service), (a gpu cluster on aws f1 instances for later).

ms-mpi is not cuda aware (https://github.com/microsoft/Microsoft-MPI/issues/9) like open-mpi.

doesn't matter for now, no need to communicate between gpus for this program - difference would be that on cuda aware implementation we would not need to go through the host to pass buffers.

multi process service will allow overlap of kernel executions since gpu ressources are not fully utilized.

2 ways for using mpi on smallpt that come to my head:

- each process does samples/num_processes samples and then we sum them up with mpi reduce

- divide the img between the imgs and each one processes its portion and then mpi gather.

The first way is the one implemented, although my intuition tells me the second is better, since you only have to allocate H*W*3/num_procceses floats
which allows the kernels to start earlier (there won't be that much of a difference since the program is computation bound - checked with profiling).

[![img.png](https://i.postimg.cc/02Qtzj5h/img.png)](https://postimg.cc/PNgbctwM)

