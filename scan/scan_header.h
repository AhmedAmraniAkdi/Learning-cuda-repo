#ifndef SCAN_HEADER_H
#define SCAN_HEADER_H

#define BLOCKDIM 1024
#define LOG2_BLOCKDIM 10

void scan(float* d_input, float* d_output, int arr_size);

#endif