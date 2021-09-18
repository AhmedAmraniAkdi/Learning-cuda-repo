Sorting array of floats on Cuda, the idea is as follows: each block will use odd even merge sort on 512 elements, this will leave us with length/512 sorted lists that we will merge using merge Path.

References:

Merge Path - A Visually Intuitive Approach to Parallel Merging  - Oded Green, Saher Odehb , Yitzhak Birkb

Designing Efficient Sorting Algorithms for Manycore GPUs - Nadathur Satish, Mark Harris, Michael Garland

Fast in-place, comparison-based sorting with CUDA: a study with bitonic sort - Hagen Peters, Ole Schulz-Hildebrandt, Norbert Luttenberger

Sorting using bitonic network with CUDA - Ranieri Baraglia, Gabriele Capannini, Franco Maria Nardini

Comparison of parallel sorting algorithms - Darko Bozidar, Tomaz Dobravec

Odd even merge sort - The Art of Computer Programming, vol 3 (algorithm 5.2.2M)

Warpsize bitonic sort - https://on-demand.gputechconf.com/gtc/2013/presentations/S3174-Kepler-Shuffle-Tips-Tricks.pdf