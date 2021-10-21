#define PI 3.14159265358979323846
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
	if (3 != argc) {
		printf("Usage: stencil num_values num_steps\n");
		return 0;
	}
  int num_values = atoi(argv[1]);
	int num_steps = atoi(argv[2]);
	int rank, size;
	double execution_time, total_start_time, start_time, max_time, total_time;

	MPI_Status status;
	MPI_Request request;

  MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size); /* Get the number of processors */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* Get my number                */
	int chunk = num_values/size; // number of values for every process

	// Stencil values
	const int STENCIL_WIDTH = 5;
	const int EXTENT = STENCIL_WIDTH/2;
	double h = 2.0*PI/num_values;
	const double STENCIL[] = {1.0/(12*h), -8.0/(12*h), 0.0, 8.0/(12*h), -1.0/(12*h)};

	// Check assumptions
	if(num_values%size != 0) {
		if(rank==0) printf("ERROR: num_values must be divisible by number of processes!\n");
		exit(0);
	} else if(chunk < STENCIL_WIDTH){
		if(rank==0) printf("ERROR: Too many processors for given num_values!\n");
		exit(1);
	}

	// Allocate data for input and output
	double *input=(double*)malloc(num_values*sizeof(double));
	double *output =(double*)malloc(num_values*sizeof(double));
	double *local_input =(double *)malloc((chunk+EXTENT*2)*sizeof(double));
	double *local_output =(double *)malloc(chunk*sizeof(double));

	if(rank==0){
		// Generate values for stencil operation
    for (int i=0; i<num_values; i++) input[i]=sin(h*i);

		// Print input values
		/*printf("input:\n");
		for(int i=0; i<num_values; i++){
			printf("%f ", input[i]);
		}
		printf("\n");*/

		// Write input values to file
		/*
  	FILE *file1=fopen("input.txt","w");
  	for (int i = 0; i < num_values; i++)
    	fprintf(file1, "%f \n", input[i]);
  	fclose(file1);*/

		total_start_time = MPI_Wtime();
	}

	MPI_Scatter(&input[0], chunk, MPI_DOUBLE, &local_input[EXTENT], chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Start timer
	start_time = MPI_Wtime();

	// Repeatedly apply stencil
	for (int s=0; s<num_steps; s++) {
		if(rank==0){
			MPI_Isend(&local_input[chunk+EXTENT-2], 2, MPI_DOUBLE, 1, 300+1, MPI_COMM_WORLD, &request);
			MPI_Isend(&local_input[2], 2, MPI_DOUBLE, size-1, 100+(size-1), MPI_COMM_WORLD, &request);

			MPI_Irecv(&local_input[chunk+EXTENT], 2, MPI_DOUBLE, 1, 100+rank, MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);

			MPI_Irecv(&local_input[0], 2, MPI_DOUBLE, size-1, 300+0, MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);			
		} else if(rank==size-1){
			MPI_Isend(&local_input[chunk+EXTENT-2], 2, MPI_DOUBLE, 0, 300+0, MPI_COMM_WORLD, &request);
			MPI_Isend(&local_input[2], 2, MPI_DOUBLE, rank-1, 100+(rank-1), MPI_COMM_WORLD, &request);

			MPI_Irecv(&local_input[chunk+EXTENT], 2, MPI_DOUBLE, 0, 100+rank, MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);
			MPI_Irecv(&local_input[0], 2, MPI_DOUBLE, rank-1, 300+rank, MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);
		} else {
			MPI_Isend(&local_input[2], 2, MPI_DOUBLE, rank-1, 100+(rank-1), MPI_COMM_WORLD, &request);
			MPI_Isend(&local_input[chunk+EXTENT-2], 2, MPI_DOUBLE, rank+1, 300+(rank+1), MPI_COMM_WORLD, &request);

			MPI_Irecv(&local_input[0], 2, MPI_DOUBLE, rank-1, 300+rank, MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);
			MPI_Irecv(&local_input[chunk+EXTENT], 2, MPI_DOUBLE, rank+1, 100+rank, MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);
		}

    // Apply stencil
		for (int i=EXTENT; i<chunk+EXTENT; i++) {
			double result = 0;
			for (int j=0; j<STENCIL_WIDTH; j++) {
				int index = i - EXTENT + j;
				result += STENCIL[j] * local_input[index];
			}
			local_output[i-EXTENT] = result;
		}

		MPI_Barrier(MPI_COMM_WORLD); // synchronize processes

		// Swap input and output
		if (s < num_steps-1) {
			for(int i=0; i<chunk; i++){
				local_input[i+EXTENT] = local_output[i];
			}
		} else {
			double execution_time = MPI_Wtime()-start_time; // stop timer
			MPI_Gather(&local_output[0], chunk, MPI_DOUBLE, &output[0], chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			if (rank == 0) total_time = MPI_Wtime()-total_start_time; // stop timer
		}
	}

	// Print output values
	/*if(rank==0){
		printf("output:\n");
		for(int i=0; i<num_values; i++){
			printf("%f ", output[i]);
		}
		printf("\n");
	}*/

	execution_time = MPI_Wtime()-start_time; // stop timer

	// Find max time among processors
	MPI_Reduce(&execution_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); 

	// Display timing results
	if(rank==0){ 
		printf("Maximum stencil application time: %f\n", max_time);
		printf("Total time: %f\n", total_time);
	}

	/*
	// Write to file
  FILE *file=fopen("output.txt","w");
  for (int i = 0; i < num_values; i++)
    fprintf(file, "%f \n", output[i]);
  fclose(file);
  */

	// Clean up
  free(input);
	free(output);
	free(local_input);
	free(local_output);
  MPI_Finalize();
	return 0;
}
