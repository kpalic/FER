#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <mpi.h>

#define BOARD_HEIGHT 6
#define BOARD_WIDTH 7
#define PLAYER 1
#define AI 2

#define TASK_DEPTH = 5
#define SEARCH_DEPTH = 6

struct Move
{
    int colNumberAI;
    int colNumberPlayer;
};


void printBoard(char board[BOARD_HEIGHT][BOARD_WIDTH]) {
    int i, j;

    // Print the board
    for (i = 0; i < BOARD_HEIGHT; i++) {
        for (j = 0; j < BOARD_WIDTH; j++) {
            printf("| %c ", board[i][j]);
        }
        printf("|\n");
    }
    for (j = 0; j < BOARD_WIDTH; j++) {
        printf("  %d ", j);
    }
    printf("\n");
}

bool validateMove(char board[BOARD_HEIGHT][BOARD_WIDTH], int move) {
    if (move < 0 || move >= BOARD_WIDTH) {
        return false;
    }
    if (board[0][move] != ' ') {
        return false;
    }
    return true;
}

int makeMove(int player, char board[BOARD_HEIGHT][BOARD_WIDTH], int colNumber) {
    char playerMove = player == 1 ? 'X' : 'O';

    // Make the move
    for (int i = BOARD_HEIGHT - 1; i >= 0; i--) {
        if (board[i][colNumber] == ' ') {
            board[i][colNumber] = playerMove;
            break;
        }
    }

    return colNumber;
}

bool checkWin(int colNumber, char board[BOARD_HEIGHT][BOARD_WIDTH], int player) {
    int rowNumber = BOARD_HEIGHT;
    for (int i = 0; i < BOARD_HEIGHT; i++) {
        if (board[i][colNumber] != ' ') {
            rowNumber = i;
            break;
        }
    }

    // Check horizontal
    int count = 0;
    for (int i = 0; i < BOARD_WIDTH; i++) {
        if (board[rowNumber][i] == (player == 1 ? 'X' : 'O')) {
            count++;
            if (count == 4) {
                printf(player == 1 ? "Player wins!\n" : "AI wins!\n");
                return true;
            }
        } else {
            count = 0;
        }
    }

    // Check vertical
    count = 0;
    for (int i = 0; i < BOARD_HEIGHT; i++) {
        if (board[i][colNumber] == (player == 1 ? 'X' : 'O')) {
            count++;
            if (count == 4) {
                printf(player == 1 ? "Player wins!\n" : "AI wins!\n");
                return true;
            }
        } else {
            count = 0;
        }
    }

    // Check left diagonal containing the last move
    count = 0;
    for (int i = rowNumber - 3, j = colNumber - 3; i <= rowNumber + 3 && j <= colNumber + 3; i++, j++) {
        if (i >= 0 && i < BOARD_HEIGHT && j >= 0 && j < BOARD_WIDTH) {
            if (board[i][j] == (player == 1 ? 'X' : 'O')) {
                count++;
                if (count == 4) {
                    printf(player == 1 ? "Player wins!\n" : "AI wins!\n");
                    return true;
                }
            } else {
                count = 0;
            }
        }
    }

    // Check right diagonal containing the last move
    count = 0;
    for (int i = rowNumber - 3, j = colNumber + 3; i <= rowNumber + 3 && j >= colNumber - 3; i++, j--) {
        if (i >= 0 && i < BOARD_HEIGHT && j >= 0 && j < BOARD_WIDTH) {
            if (board[i][j] == (player == 1 ? 'X' : 'O')) {
                count++;
                if (count == 4) {
                    printf(player == 1 ? "Player wins!\n" : "AI wins!\n");
                    return true;
                }
            } else {
                count = 0;
            }
        }
    }
    return false;
}


void worker() {

}

void master() {
    char board[BOARD_HEIGHT][BOARD_WIDTH];
    for (int i = 0; i < BOARD_HEIGHT; i++) {
        for (int j = 0; j < BOARD_WIDTH; j++) {
            board[i][j] = ' ';
        }
    }
    bool gameOver = false;
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("Master starting with %d processes\n", size);
    int comm = MPI_COMM_WORLD;
    bool unemployedWorkers[size - 1];
    for (int i = 0; i < size - 1; i++) {
        unemployedWorkers[i] = true;
    }

    while (!gameOver) {
        int colNumber;
        printBoard(board);
        printf("Enter a column number: ");
        scanf("%d", &colNumber);

        while (!validateMove(board, colNumber)) {
                printf("Invalid move. Enter a column number: ");
                scanf("%d", &colNumber);
        }

        makeMove(PLAYER, board, colNumber);
        if (checkWin(colNumber, board, PLAYER)) {
            gameOver = true;
            break;
        }


        


    }

}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        master();
    } else {
        worker();
    }

    MPI_Finalize();
    return 0;
}