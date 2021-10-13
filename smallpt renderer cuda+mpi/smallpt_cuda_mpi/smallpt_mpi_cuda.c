#include <stdio.h>
#include <mpi.h>
#include "smallpt_cuda_mpi_h_.h"

#define H 768
#define W 1024


int main()
{
	int my_rank;
	int world_size;

	MPI_Init(NULL, NULL);

	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	printf("Hello World from process %d out of %d processes\n", my_rank, world_size);

	printf("starting cuda function\n");

    float *h_img = (float *)malloc(3 * H * W * sizeof(float));

    smallpt_main(my_rank, h_img);

    float *sendbuf,*recvbuf;
    if (my_rank == 0) {
        sendbuf = MPI_IN_PLACE; recvbuf = h_img;
    } else {
        sendbuf = h_img; recvbuf = MPI_IN_PLACE;
    }
    MPI_Reduce(sendbuf, recvbuf, W*H*3, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    //
    if(my_rank == 0){
        FILE *f = fopen("image_cuda_mpi.ppm", "w");         // Write image to PPM file.
        fprintf(f, "P3\n%d %d\n%d\n", W, H, 255);
        for (int i = 0; i < W*H*3; i = i + 3)
            fprintf(f, "%d %d %d ", h_img[i + 0], h_img[i + 1], h_img[i + 2]);

    }
    free(h_img);

	MPI_Finalize();
	return 0;
}