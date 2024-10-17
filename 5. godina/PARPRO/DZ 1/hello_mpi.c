#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    // Inicijalizacija MPI okoline
    MPI_Init(&argc, &argv);

    // Dohvaćanje broja procesa
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Dohvaćanje ranga procesa
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Ispis poruke
    printf("Hello world from processor %d out of %d\n", world_rank, world_size);

    // Završetak MPI okoline
    MPI_Finalize();

    return 0;
}
