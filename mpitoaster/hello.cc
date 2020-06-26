#include <mpi.h>
 
int main(int argc, char **argv)
{
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 1) {
        char *d = 0;
        /* This will cause a seg fault */
        *d = 3;
    }
 
    MPI_Finalize();
    return 0;
}
