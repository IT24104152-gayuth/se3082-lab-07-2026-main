#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0)
            printf("N must be evenly divisible by number of processes.\n");

        MPI_Finalize();
        return 1;
    }

    int chunk_size = N / size;

    int *array = NULL;

    /* Only root stores and initializes the full array */
    if (rank == 0) {
        array = malloc(N * sizeof(int));

        for (int i = 0; i < N; i++) {
            array[i] = i + 1;
        }

        printf("Root filled array with values 1 to %d\n", N);
    }

    /* Every process receives one local chunk */
    int *local_chunk = malloc(chunk_size * sizeof(int));

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    /* Divide the array among the processes */
    MPI_Scatter(
        array,
        chunk_size,
        MPI_INT,
        local_chunk,
        chunk_size,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    /* Calculate local sum */
    long long local_sum = 0;

    for (int i = 0; i < chunk_size; i++) {
        local_sum += local_chunk[i];
    }

    /*
     * Allreduce adds all local sums and gives
     * the final result to EVERY process.
     */
    long long total_sum = 0;

    MPI_Allreduce(
        &local_sum,
        &total_sum,
        1,
        MPI_LONG_LONG,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    /* Every process can use total_sum */
    double percentage =
        ((double)local_sum / (double)total_sum) * 100.0;

    printf(
        "Rank %d: local_sum = %lld, total_sum = %lld, contribution = %.2f%%\n",
        rank,
        local_sum,
        total_sum,
        percentage
    );

    /* Verification only on root */
    if (rank == 0) {
        double elapsed = MPI_Wtime() - start;

        long long expected =
            (long long)N * (N + 1) / 2;

        printf("\n[Allreduce] Total sum = %lld\n", total_sum);
        printf("[Allreduce] Expected  = %lld\n", expected);
        printf("[Allreduce] Correct?  = %s\n",
               total_sum == expected ? "YES" : "NO");
        printf("[Allreduce] Time      = %.6f sec\n", elapsed);
    }

    free(local_chunk);

    if (rank == 0) {
        free(array);
    }

    MPI_Finalize();

    return 0;
}