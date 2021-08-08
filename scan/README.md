CPU ~250ms

GPU my scan ~12ms - profiling says need overlapping memory io and compute -> streams // bad latency caused by the syncthreads and if elses?
GPU thrust ~4ms - i think this is a win.