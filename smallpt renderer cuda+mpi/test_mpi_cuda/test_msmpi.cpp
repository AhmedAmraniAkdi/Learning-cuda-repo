// A simple MPI code printing a message by each MPI rank

// works

#include <iostream>
#include <mpi.h>
#include "device_query.h"


int main()
{
	int my_rank;
	int world_size;

	MPI_Init(NULL, NULL);

	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	std::cout << "Hello World from process " << my_rank << " out of " << world_size << " processes!!!" << std::endl;

	std::cout<< "starting device query" << std::endl;

	device_query(my_rank);

	MPI_Finalize();
	return 0;
}

/*

cl test_msmpi.cpp /EHsc /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include\x64" /link /machine:x64 /dynamicbase "msmpi.lib" /libpath:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" 

*/