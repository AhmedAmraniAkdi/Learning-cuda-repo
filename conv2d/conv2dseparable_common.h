#ifndef CONVOLUTIONSEPARABLE_COMMON_H
#define CONVOLUTIONSEPARABLE_COMMON_H

#define KERNELRADIUS 8 // 8x2 + 1
#define BLOCKDIM 64
#define STEP 4

extern void processing( // extern "C" ??
    float* h_input,
    float *h_output,
    float *h_kernel,
    int img_w, 
    int img_h,
    int kernelradius
);

#endif