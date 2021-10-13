#include <iostream>
#include <mpi.h>
#include "smallpt_cuda_mpi_h_.h"


int main()
{
	int my_rank;
	int world_size;

	MPI_Init(NULL, NULL);

	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	std::cout << "Hello World from process " << my_rank << " out of " << world_size << " processes!!!" << std::endl;

	std::cout<< "starting device query" << std::endl;

	MPI_Finalize();
	return 0;
}