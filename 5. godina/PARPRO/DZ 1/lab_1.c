#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <windows.h>
#include <time.h>

#define WANTS 0
#define GIVES 1

struct fork
{
    bool isFree;
    int currentUser;
    bool isClean;
};


void Philosopher(int rang, int numPhilosophers, struct fork *forks)
{
    MPI_Status status;
    srand(time(NULL) + rang);
    struct fork faultyData[numPhilosophers];
    while (true)
    {

        //misli
        int flag = 0;
        int flagLeftWanted = 0;
        int flagRightWanted = 0;
        int randomNumber = (rand() % 10) + 1;
        printf("Filozof %d misli %d sekundi\n", rang, randomNumber);
        fflush(stdout);
        for (int i = 0; i < randomNumber; i++)
        {
            Sleep(1000);
            // postoji li zahtjev za lijevom vilicom
            int tempFlagLeftWanted = 0;
            MPI_Iprobe((rang + numPhilosophers - 1) % numPhilosophers, WANTS, MPI_COMM_WORLD, &tempFlagLeftWanted, &status);
            if (tempFlagLeftWanted) {
                MPI_Recv(&faultyData[(rang + numPhilosophers - 1) % numPhilosophers], sizeof(faultyData[(rang + numPhilosophers - 1) % numPhilosophers]), MPI_BYTE, (rang + numPhilosophers - 1) % numPhilosophers, WANTS, MPI_COMM_WORLD, &status);
                // printf("Filozof %d primio zahtjev od lijevog susjeda(%d)\n", rang, status.MPI_SOURCE);
                // fflush(stdout);
                if (forks[rang].isClean == false) {
                    // ocisti i posalji vilicu
                    forks[rang].isClean = true;
                    forks[rang].currentUser = (rang + numPhilosophers - 1) % numPhilosophers;
                    forks[rang].isFree = false;
                    // printf("Filozof %d poslao vilicu lijevom susjedu(%d)\n", rang, (rang + numPhilosophers - 1) % numPhilosophers);
                    // fflush(stdout);
                    MPI_Send(&forks[rang], sizeof(forks[rang]), MPI_BYTE, (rang + numPhilosophers - 1) % numPhilosophers, GIVES, MPI_COMM_WORLD);
                    flagLeftWanted = 0;
                }
                else {
                    // zapamti zahtjev, nemoj odmah poslati vilicu
                    // printf("Filozof %d zapamtio zahtjev lijevog susjeda(%d)\n", rang, status.MPI_SOURCE);
                    // fflush(stdout);
                    flagLeftWanted = 1;
                }
            }

            // postoji li zahtjev za desnom vilicom 
            int tempFlagRightWanted = 0;
            MPI_Iprobe((rang + 1) % numPhilosophers, WANTS, MPI_COMM_WORLD, &tempFlagRightWanted, &status);
            if (tempFlagRightWanted) {
                MPI_Recv(&faultyData[(rang + 1) % numPhilosophers], sizeof(faultyData[(rang + 1) % numPhilosophers]), MPI_BYTE, (rang + 1) % numPhilosophers, WANTS, MPI_COMM_WORLD, &status);
                // printf("Filozof %d primio zahtjev od desnog susjeda(%d)\n", rang, status.MPI_SOURCE);
                // fflush(stdout);
                if (forks[(rang+1) % numPhilosophers].isClean == false) {
                    // ocisti i posalji vilicu
                    forks[(rang + 1) % numPhilosophers].isClean = true;
                    forks[(rang + 1) % numPhilosophers].currentUser = (rang + 1) % numPhilosophers;
                    forks[(rang + 1) % numPhilosophers].isFree = false;
                    // printf("Filozof %d poslao vilicu desnom susjedu(%d)\n", rang, status.MPI_SOURCE);
                    // fflush(stdout);
                    MPI_Send(&forks[(rang + 1) % numPhilosophers], sizeof(forks[(rang + 1) % numPhilosophers]), MPI_BYTE, (rang + 1) % numPhilosophers, GIVES, MPI_COMM_WORLD);
                    flagRightWanted = 0;
                }
                else {
                    // zapamti zahtjev, nemoj odmah poslati vilicu
                    // printf("Filozof %d zapamtio zahtjev desnog susjeda(%d)\n", rang, status.MPI_SOURCE);
                    // fflush(stdout);
                    flagRightWanted = 1;
                }
            }

        }
        // printf("Filozof %d zavrsio razmisljanje\n", rang);
        // fflush(stdout);

        bool hasLeftFork = false;
        bool hasRightFork = false;

        if (forks[rang].isFree || forks[rang].currentUser == rang) {
            // printf("Filozof %d ima lijevu vilicu\n", rang);
            forks[rang].currentUser = rang;
            forks[rang].isFree = false;
            hasLeftFork = true;
        }

        if (forks[(rang + 1) % numPhilosophers].isFree || forks[(rang + 1) % numPhilosophers].currentUser == rang) {
            // printf("Filozof %d ima desnu vilicu\n", rang);
            forks[(rang + 1) % numPhilosophers].currentUser = rang;
            forks[(rang + 1) % numPhilosophers].isFree = false;
            hasRightFork = true;
        }

        while (hasLeftFork == false || hasRightFork == false) {
            bool leftRequested = hasLeftFork;
            bool rightRequested = hasRightFork;
            if (!hasLeftFork && !leftRequested) {
                // pošalji zahtjev za lijevom vilicom
                MPI_Send(&forks[rang], sizeof(forks[rang]), MPI_BYTE, (rang + numPhilosophers - 1) % numPhilosophers, WANTS, MPI_COMM_WORLD);
                printf("Filozof %d trazi vilicu od (%d) \n", rang, (rang + numPhilosophers - 1) % numPhilosophers);
                fflush(stdout);
                // printf("Stanja vilica kako ih vidi filozof %d\n", rang);
                // for (int i = 0; i < numPhilosophers; i++) {
                //     printf("Vilica %d\n", i);
                //     printf("isFree: %d\n", forks[i].isFree);
                //     printf("currentUser: %d\n", forks[i].currentUser);
                //     printf("isClean: %d\n\n", forks[i].isClean);
                //     fflush(stdout);
                // }
                leftRequested = true;
            } 
            if (!hasRightFork && !rightRequested) {
                // pošalji zahtjev za desnom vilicom
                MPI_Send(&forks[(rang + 1) % numPhilosophers], sizeof(forks[(rang + 1) % numPhilosophers]), MPI_BYTE, (rang + 1) % numPhilosophers, WANTS, MPI_COMM_WORLD);
                printf("Filozof %d trazi vilicu od (%d)\n", rang, (rang + 1) % numPhilosophers);
                fflush(stdout);
                // printf("Stanja vilica kako ih vidi filozof %d\n", rang);
                // for (int i = 0; i < numPhilosophers; i++) {
                //     printf("Vilica %d\n", i);
                //     printf("isFree: %d\n", forks[i].isFree);
                //     printf("currentUser: %d\n", forks[i].currentUser);
                //     printf("isClean: %d\n\n", forks[i].isClean);
                //     fflush(stdout);
                // }
                rightRequested = true;
            }
            bool resolvedRequest = false;
            int flag = 0;
            while (!resolvedRequest) {
                flag = 0;
                MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
                Sleep(1000);
                if (flag) {
                    if (status.MPI_TAG == GIVES) {
                        // dobili smo vilicu, koju?
                        if (status.MPI_SOURCE == (rang + numPhilosophers - 1) % numPhilosophers) {
                            // dobili smo lijevu vilicu
                            MPI_Recv(&forks[rang], sizeof(forks[rang]), MPI_BYTE, (rang + numPhilosophers - 1) % numPhilosophers, GIVES, MPI_COMM_WORLD, &status);
                            printf("Filozof %d dobio vilicu od lijevog susjeda\n", rang);
                            fflush(stdout);
                            forks[rang].currentUser = rang;
                            forks[rang].isFree = false;
                            hasLeftFork = true;
                            resolvedRequest = true;
                        }
                        else {
                            // dobili smo desnu vilicu
                            MPI_Recv(&forks[(rang + 1) % numPhilosophers], sizeof(forks[(rang + 1) % numPhilosophers]), MPI_BYTE, (rang + 1) % numPhilosophers, GIVES, MPI_COMM_WORLD, &status);
                            printf("Filozof %d dobio vilicu od desnog susjeda\n", rang);
                            fflush(stdout);
                            forks[(rang + 1) % numPhilosophers].currentUser = rang;
                            forks[(rang + 1) % numPhilosophers].isFree = false;
                            hasRightFork = true;
                            resolvedRequest = true;
                        }
                    }
                    else {
                        // dobili smo zahtjev za vilicu, koju?
                        if (status.MPI_SOURCE == (rang + numPhilosophers - 1) % numPhilosophers) {
                            // zahtjev za lijevom vilicom
                            MPI_Recv(&faultyData[rang], sizeof(faultyData[rang]), MPI_BYTE, (rang + numPhilosophers - 1) % numPhilosophers, WANTS, MPI_COMM_WORLD, &status);
                            if (forks[rang].isClean == false) {
                                // posalji vilicu
                                printf("Filozof %d poslao lijevu vilicu \n", rang);
                                fflush(stdout);
                                forks[rang].isClean = true;
                                forks[rang].currentUser = (rang + numPhilosophers - 1) % numPhilosophers;
                                MPI_Send(&forks[rang], sizeof(forks[rang]), MPI_BYTE, (rang + numPhilosophers - 1) % numPhilosophers, GIVES, MPI_COMM_WORLD);
                                resolvedRequest = true;
                                hasLeftFork = false;
                                leftRequested = false;
                            }
                            else {
                                // zabiljezi zahtjev
                                printf("Filozof %d zabiljezio zahtjev za lijevom vilicom\n", rang);
                                fflush(stdout);
                                flagLeftWanted = 1;
                            }
                        }
                        else {
                            // zahtjev za desnom vilicom

                            MPI_Recv(&faultyData[(rang + 1) % numPhilosophers], sizeof(faultyData[(rang + 1) % numPhilosophers]), MPI_BYTE, (rang + 1) % numPhilosophers, WANTS, MPI_COMM_WORLD, &status);
                            if (forks[(rang + 1) % numPhilosophers].isClean == false) {
                                // posalji vilicu
                                printf("Filozof %d poslao desnu vilicu \n", rang);
                                fflush(stdout);
                                forks[(rang + 1) % numPhilosophers].isClean = true;
                                forks[(rang + 1) % numPhilosophers].currentUser = (rang + 1) % numPhilosophers;
                                MPI_Send(&forks[(rang + 1) % numPhilosophers], sizeof(forks[(rang + 1) % numPhilosophers]), MPI_BYTE, (rang + 1) % numPhilosophers, GIVES, MPI_COMM_WORLD);
                                resolvedRequest = true;
                                hasRightFork = false;
                                rightRequested = false;
                            }
                            else {
                                // zabiljezi zahtjev
                                printf("Filozof %d zabiljezio zahtjev za desnom vilicom\n", rang);
                                fflush(stdout);
                                flagRightWanted = 1;
                            }
                        }
                    }
                }
            }
        } // imamo obje vilice

        // jedi
        printf("Filozof %d jede\n", rang);
        fflush(stdout);
        randomNumber = (rand() % 10) + 1;
        Sleep(randomNumber * 1000);
        printf("Filozof %d zavrsio jelo\n", rang);
        fflush(stdout);
        forks[rang].isClean = false;
        forks[(rang + 1) % numPhilosophers].isClean = false;

        // oslobodi vilice
        // forks[rang].isFree = false;
        // forks[(rang + 1) % numPhilosophers].isFree = false;

        // oslobodi vilice susjedima
        // forks[rang].currentUser = (rang + numPhilosophers - 1) % numPhilosophers;
        // forks[(rang + 1) % numPhilosophers].currentUser = (rang + 1) % numPhilosophers;

    

        MPI_Status status1;
        MPI_Status status2;
        int auxFlagLeft = 0;
        int auxFlagRight = 0;
        int iter = 0;

        while (iter < 1000 && (auxFlagLeft == 0 || auxFlagRight == 0)) {
            // printf("iteracija %d\n", iter);
            // fflush(stdout);
            if(auxFlagLeft == 0) {
                MPI_Iprobe((rang + numPhilosophers - 1) % numPhilosophers, WANTS, MPI_COMM_WORLD, &auxFlagLeft, MPI_STATUS_IGNORE);
            }
            if(auxFlagRight == 0) {
                MPI_Iprobe((rang + 1) % numPhilosophers, WANTS, MPI_COMM_WORLD, &auxFlagRight, MPI_STATUS_IGNORE);
            }
            iter++;
        }

        if (auxFlagLeft) {
            MPI_Recv(&faultyData[rang], sizeof(faultyData[(rang + numPhilosophers - 1) % numPhilosophers]), MPI_BYTE, (rang + numPhilosophers - 1) % numPhilosophers, WANTS, MPI_COMM_WORLD, &status);
        }
        if(auxFlagRight) {
            MPI_Recv(&faultyData[(rang + 1) % numPhilosophers], sizeof(faultyData[(rang + 1) % numPhilosophers]), MPI_BYTE, (rang + 1) % numPhilosophers, WANTS, MPI_COMM_WORLD, &status);
        }

        // posalji vilice susjedima
        // MPI_Send(&forks[rang], sizeof(forks[rang]), MPI_BYTE, (rang + numPhilosophers - 1) % numPhilosophers, GIVES, MPI_COMM_WORLD);
        // MPI_Send(&forks[(rang + 1) % numPhilosophers], sizeof(forks[(rang + 1) % numPhilosophers]), MPI_BYTE, (rang + 1) % numPhilosophers, GIVES, MPI_COMM_WORLD);

        // printf("parametri filozofa %d\n", rang);
        // printf("flagLeftWanted: %d\n", flagLeftWanted);
        // printf("flagRightWanted: %d\n", flagRightWanted);
        // printf("auxFlagLeft: %d\n", auxFlagLeft);
        // printf("auxFlagRight: %d\n", auxFlagRight);
        // fflush(stdout);

        
        if (flagLeftWanted || auxFlagLeft) {
            forks[rang].currentUser = (rang + numPhilosophers - 1) % numPhilosophers;
            forks[rang].isFree = false;
            // printf("Filozof %d poslao lijevu vilicu lijevom susjedu(%d)\n", rang, (rang + numPhilosophers - 1) % numPhilosophers);
            MPI_Send(&forks[rang], sizeof(forks[rang]), MPI_BYTE, (rang + numPhilosophers - 1) % numPhilosophers, GIVES, MPI_COMM_WORLD);
            flagLeftWanted = 0;
        }
        else {
            // printf("Filozof %d oslobodio lijevu vilicu\n", rang);
            forks[rang].isFree = true;
            forks[rang].currentUser = -1;
        }
        if (flagRightWanted || auxFlagRight) {
            forks[(rang + 1) % numPhilosophers].currentUser = (rang + 1) % numPhilosophers;
            // printf("Filozof %d poslao desnu vilicu desnog susjedu(%d)\n", rang, (rang + 1) % numPhilosophers);
            MPI_Send(&forks[(rang + 1) % numPhilosophers], sizeof(forks[(rang + 1) % numPhilosophers]), MPI_BYTE, (rang + 1) % numPhilosophers, GIVES, MPI_COMM_WORLD);
        }
        else {
            // printf("Filozof %d oslobodio desnu vilicu\n", rang);
            forks[(rang + 1) % numPhilosophers].isFree = true;
            forks[(rang + 1) % numPhilosophers].currentUser = -1;
        }

        // printf("Nakon sto je filozof %d pojeo, vilice su:\n", rang);
        // for (int i = 0; i < numPhilosophers; i++) {
        //     printf("Vilica %d\n", i);
        //     printf("isFree: %d\n", forks[i].isFree);
        //     printf("currentUser: %d\n", forks[i].currentUser);
        //     printf("isClean: %d\n\n", forks[i].isClean);
        //     fflush(stdout);
        // }
    }
}


int main(int argc, char** argv) {
    // setvbuf(stdout, NULL, _IONBF, 0);
    // Inicijalizacija MPI okoline
    MPI_Init(&argc, &argv);

    // Dohvaćanje broja procesa
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Dohvaćanje ranga procesa
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Potrebna najmanje 2 procesa
    if (world_size < 2) {
        fprintf(stderr, "Mora biti najmanje 2 procesa %s\n", argv[0]);
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    struct fork forks[world_size];
    for (int i = 0; i < world_size; i++) {
        forks[i].isFree = false;
        forks[i].currentUser = i == 0 ? 0 : i - 1;
        forks[i].isClean = true;
    }

    // if (world_rank == 0) {
    //     for (int i = 0; i < world_size; i++) {
    //         printf("Vilica %d\n", i);
    //         printf("isFree: %d\n", forks[i].isFree);
    //         printf("currentUser: %d\n", forks[i].currentUser);
    //         printf("isClean: %d\n\n", forks[i].isClean);
    //         fflush(stdout);
    //     }
    // }

    Sleep(1000);
    for (int i = 0; i < world_size; i++) {
        if (world_rank == i) {
            Philosopher(i, world_size, forks);
        }
    }

    // Završetak MPI okoline
    MPI_Finalize();

    return 0;
}

// Proces(i)
// { 
//     misli (slucajan broj sekundi);               // ispis: mislim
//     i 'istovremeno' odgovaraj na zahtjeve!       // asinkrono, s povremenom provjerom
//     dok (nemam obje vilice) {
//         posalji zahtjev za vilicom;              // ispis: trazim vilicu (i)
//         ponavljaj {
//             cekaj poruku (bilo koju!);
//             ako je poruka odgovor na zahtjev     // dobio vilicu
//             azuriraj vilice;
//             inace ako je poruka zahtjev          // drugi traze moju vilicu
//             obradi zahtjev (odobri ili zabiljezi);
//         } dok ne dobijes trazenu vilicu;
//     }
//     jedi;                                        // ispis: jedem
//     odgovori na postojeće zahtjeve;              // ako ih je bilo
// }