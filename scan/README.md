CPU ~250ms



GPU scan_v1 ~12ms - profiling says need overlapping memory io and compute -> streams // bad latency caused by the syncthreads and if elses?

GPU thrust ~4ms - i think this is a win.

GPU scan_v2