#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define H 768
#define W 1024

extern void smallpt_main(int my_rank, float *h_img_output);

float clamp(float a, float m, float M){
    return max(m, min(a, M));
}

int toInt(float x){ 
    return (int)(pow(clamp(x, 0.0, 1.0), 1 / 2.2) * 255 + .5); 
}


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
        for (int i = 0; i < W*H*3; i = i + 3){
            fprintf(f, "%d %d %d ", toInt(*(h_img + i + 0)), toInt(*(h_img + i + 1)), toInt(*(h_img + i + 2)));
        }

    }
    free(h_img);

	MPI_Finalize();
	return 0;
}