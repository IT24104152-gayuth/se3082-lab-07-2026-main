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

    /* Only root allocates and initializes the full array */
    if (rank == 0) {

        array = malloc(N * sizeof(int));

        for (int i = 0; i < N; i++) {
            array[i] = i + 1;
        }

        printf("Root filled array with values 1 to %d\n", N);
    }

    /* Every process stores its own chunk */
    int *local_chunk =
        malloc(chunk_size * sizeof(int));

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    /* Distribute array chunks */
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
     * Prefix reduction.
     *
     * Rank 0 gets: local_sum0
     * Rank 1 gets: local_sum0 + local_sum1
     * Rank 2 gets: local_sum0 + local_sum1 + local_sum2
     * ...
     */
    long long prefix_sum = 0;

    MPI_Scan(
        &local_sum,
        &prefix_sum,
        1,
        MPI_LONG_LONG,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    long long sum_before_me =
        prefix_sum - local_sum;

    printf(
        "Rank %d: local_sum = %lld, prefix_sum = %lld, sum_before_me = %lld\n",
        rank,
        local_sum,
        prefix_sum,
        sum_before_me
    );

    /*
     * The last rank's prefix_sum is the sum
     * of the complete array.
     */
    if (rank == size - 1) {

        double elapsed =
            MPI_Wtime() - start;

        long long expected =
            (long long)N * (N + 1) / 2;

        printf("\n[Scan] Total sum = %lld\n",
               prefix_sum);

        printf("[Scan] Expected  = %lld\n",
               expected);

        printf("[Scan] Correct?  = %s\n",
               prefix_sum == expected ? "YES" : "NO");

        printf("[Scan] Time      = %.6f sec\n",
               elapsed);
    }

    free(local_chunk);

    if (rank == 0) {
        free(array);
    }

    MPI_Finalize();

    return 0;
}