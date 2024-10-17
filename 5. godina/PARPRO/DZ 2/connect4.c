#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROWS 6
#define COLUMNS 7
#define EMPTY 0
#define PLAYER 1
#define AI 2
#define DUBINA_ZADATAKA 5
#define DUBINA_PRETRAZIVANJA 6

void odigraj_potez(int ploca[ROWS][COLUMNS], int column, int player, int *row_result) {
    for (int row = ROWS - 1; row >= 0; row--) {
        if (ploca[row][column] == EMPTY) {
            ploca[row][column] = player;
            *row_result = row;
            return;
        }
    }
    *row_result = -1; // Indicating invalid move
}

int evaluate_board_job(int board[ROWS][COLUMNS]) {
    int is_player_winner(int player) {
        for (int row = 3; row < ROWS; row++) {
            for (int column = 0; column < COLUMNS - 3; column++) {
                if (board[row][column] == player && board[row - 1][column + 1] == player && 
                    board[row - 2][column + 2] == player && board[row - 3][column + 3] == player) {
                    return 1;
                }
            }
        }

        for (int row = 0; row < ROWS - 3; row++) {
            for (int column = 0; column < COLUMNS; column++) {
                if (board[row][column] == player && board[row + 1][column] == player && 
                    board[row + 2][column] == player && board[row + 3][column] == player) {
                    return 1;
                }
            }
        }

        for (int row = 0; row < ROWS - 3; row++) {
            for (int column = 0; column < COLUMNS - 3; column++) {
                if (board[row][column] == player && board[row + 1][column + 1] == player && 
                    board[row + 2][column + 2] == player && board[row + 3][column + 3] == player) {
                    return 1;
                }
            }
        }

        for (int row = 0; row < ROWS; row++) {
            for (int column = 0; column < COLUMNS - 3; column++) {
                if (board[row][column] == player && board[row][column + 1] == player && 
                    board[row][column + 2] == player && board[row][column + 3] == player) {
                    return 1;
                }
            }
        }

        return 0;
    }

    if (is_player_winner(PLAYER)) {
        return -1;
    } else if (is_player_winner(AI)) {
        return 1;
    } else {
        return 0;
    }
}

void print_board(int ploca[ROWS][COLUMNS]) {
    for (int row = 0; row < ROWS; row++) {
        for (int col = 0; col < COLUMNS; col++) {
            printf("%d ", ploca[row][col]);
        }
        printf("\n");
    }
    printf("\n");
    fflush(stdout);
}

void generiraj_poslove(int board[ROWS][COLUMNS], int broj_razina, int current_path[], int path_len, int jobs[][DUBINA_ZADATAKA], int *job_count) {
    if (broj_razina == 0) {
        memcpy(jobs[*job_count], current_path, path_len * sizeof(int));
        (*job_count)++;
        return;
    }

    for (int column = 0; column < COLUMNS; column++) {
        int next_board[ROWS][COLUMNS];
        memcpy(next_board, board, ROWS * COLUMNS * sizeof(int));
        int player = (path_len % 2 == 0) ? AI : PLAYER;
        int row_result;
        odigraj_potez(next_board, column, player, &row_result);
        if (row_result != -1) {
            current_path[path_len] = column;
            generiraj_poslove(next_board, broj_razina - 1, current_path, path_len + 1, jobs, job_count);
        }
    }
}

void calculate_scores(int current_result[][DUBINA_ZADATAKA], int scores[], int result_len, int broj_razina, int is_ai_turn) {
    if (broj_razina == 0) {
        return;
    }

    int parent_results[result_len][DUBINA_ZADATAKA];
    int parent_scores[result_len];
    int parent_len = 0;

    for (int i = 0; i < result_len; i++) {
        int parent_path_len = DUBINA_ZADATAKA - broj_razina + 1;
        memcpy(parent_results[parent_len], current_result[i], parent_path_len * sizeof(int));
        parent_scores[parent_len] = scores[i];
        parent_len++;
    }

    for (int i = 0; i < parent_len; i++) {
        if (is_ai_turn) {
            if (parent_scores[i] == -1) {
                scores[i] = -1;
            } else {
                scores[i] = 0; // Calculate average score
            }
        } else {
            if (parent_scores[i] == 1) {
                scores[i] = 1;
            } else if (parent_scores[i] == -1) {
                scores[i] = -1;
            } else {
                scores[i] = 0; // Calculate average score
            }
        }
    }

    calculate_scores(parent_results, scores, parent_len, broj_razina - 1, !is_ai_turn);
}

void worker() {
    MPI_Status status;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("Worker %d starting\n", rank);
    fflush(stdout);
    MPI_Send(NULL, 0, MPI_INT, 0, 0, MPI_COMM_WORLD); // initial job request

    while (1) {
        int message[ROWS * COLUMNS + DUBINA_ZADATAKA];
        MPI_Recv(message, ROWS * COLUMNS + DUBINA_ZADATAKA, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == 3) {
            printf("Worker %d terminating\n", rank);
            fflush(stdout);
            break;
        }

        int board[ROWS][COLUMNS];
        memcpy(board, message, ROWS * COLUMNS * sizeof(int));

        int pathToTaskNode[DUBINA_ZADATAKA];
        memcpy(pathToTaskNode, &message[ROWS * COLUMNS], DUBINA_ZADATAKA * sizeof(int));

        int current_player = AI;
        for (int i = 0; i < DUBINA_ZADATAKA; i++) {
            if (pathToTaskNode[i] != -1) {
                odigraj_potez(board, pathToTaskNode[i], current_player, &current_player);
                current_player = (current_player == AI) ? PLAYER : AI;
            }
        }

        int worker_jobs[100][DUBINA_ZADATAKA];
        int worker_job_count = 0;
        int current_path[DUBINA_ZADATAKA] = {0};

        generiraj_poslove(board, DUBINA_PRETRAZIVANJA - DUBINA_ZADATAKA, current_path, 0, worker_jobs, &worker_job_count);

        if (worker_job_count == 0) {
            int result = evaluate_board_job(board);
            int result_message[DUBINA_ZADATAKA + 1];
            memcpy(result_message, pathToTaskNode, DUBINA_ZADATAKA * sizeof(int));
            result_message[DUBINA_ZADATAKA] = result;
            MPI_Send(result_message, DUBINA_ZADATAKA + 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        } else {
            int worker_results[100][DUBINA_ZADATAKA];
            int worker_scores[100];
            int worker_result_len = 0;

            for (int i = 0; i < worker_job_count; i++) {
                int temp_board[ROWS][COLUMNS];
                memcpy(temp_board, board, ROWS * COLUMNS * sizeof(int));

                int temp_path[DUBINA_ZADATAKA];
                memcpy(temp_path, worker_jobs[i], DUBINA_ZADATAKA * sizeof(int));

                int temp_current_player = current_player;
                for (int j = 0; j < DUBINA_ZADATAKA; j++) {
                    if (temp_path[j] != -1) {
                        odigraj_potez(temp_board, temp_path[j], temp_current_player, &temp_current_player);
                        temp_current_player = (temp_current_player == AI) ? PLAYER : AI;
                    }
                }

                worker_scores[worker_result_len] = evaluate_board_job(temp_board);
                memcpy(worker_results[worker_result_len], worker_jobs[i], DUBINA_ZADATAKA * sizeof(int));
                worker_result_len++;
            }

            int is_ai_turn = (DUBINA_PRETRAZIVANJA % 2 == 0);

            calculate_scores(worker_results, worker_scores, worker_result_len, DUBINA_PRETRAZIVANJA - DUBINA_ZADATAKA, is_ai_turn);

            int score_to_send = worker_scores[0];
            int path_to_send[DUBINA_ZADATAKA];
            memcpy(path_to_send, worker_results[0], DUBINA_ZADATAKA * sizeof(int));

            int result_message[DUBINA_ZADATAKA + 1];
            memcpy(result_message, pathToTaskNode, DUBINA_ZADATAKA * sizeof(int));
            result_message[DUBINA_ZADATAKA] = score_to_send;
            MPI_Send(result_message, DUBINA_ZADATAKA + 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        }
    }
}

void master() {
    int board[ROWS][COLUMNS] = {0};
    int game_over = 0;
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("Master starting with %d processes\n", size);
    fflush(stdout);

    while (!game_over) {
        print_board(board);
        int column;
        printf("Select a column (0-6): ");
        fflush(stdout);
        scanf("%d", &column);

        int row_result;
        odigraj_potez(board, column, PLAYER, &row_result);

        if (evaluate_board_job(board) == -1) {
            print_board(board);
            printf("Player wins\n");
            fflush(stdout);
            game_over = 1;
            break;
        }

        int jobs[100][DUBINA_ZADATAKA];
        int job_count = 0;
        int current_path[DUBINA_ZADATAKA] = {0};
        generiraj_poslove(board, DUBINA_ZADATAKA, current_path, 0, jobs, &job_count);

        int results[100];
        int result_count = 0;

        MPI_Status status;
        while (result_count < job_count) {
            for (int i = 0; i < job_count; i++) {
                int message[ROWS * COLUMNS + DUBINA_ZADATAKA];
                memcpy(message, board, ROWS * COLUMNS * sizeof(int));
                memcpy(&message[ROWS * COLUMNS], jobs[i], DUBINA_ZADATAKA * sizeof(int));
                MPI_Send(message, ROWS * COLUMNS + DUBINA_ZADATAKA, MPI_INT, i % (size - 1) + 1, 1, MPI_COMM_WORLD);
            }

            for (int i = 0; i < job_count; i++) {
                int result_message[DUBINA_ZADATAKA + 1];
                MPI_Recv(result_message, DUBINA_ZADATAKA + 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                int path[DUBINA_ZADATAKA];
                memcpy(path, result_message, DUBINA_ZADATAKA * sizeof(int));
                results[result_count] = result_message[DUBINA_ZADATAKA];
                result_count++;
            }
        }

        int is_ai_turn = (DUBINA_ZADATAKA % 2 == 0);
        int final_results[100];
        calculate_scores(jobs, results, result_count, DUBINA_ZADATAKA - 1, is_ai_turn);

        int best_score = -2;
        int best_column = -1;
        for (int i = 0; i < result_count; i++) {
            if (results[i] > best_score) {
                best_score = results[i];
                best_column = jobs[i][0];
            }
        }

        if (best_column != -1) {
            odigraj_potez(board, best_column, AI, &row_result);
            if (evaluate_board_job(board) == 1) {
                print_board(board);
                printf("AI wins\n");
                fflush(stdout);
                game_over = 1;
                break;
            }
        }

        printf("Round completed\n");
        fflush(stdout);
    }

    for (int i = 1; i < size; i++) {
        MPI_Send(NULL, 0, MPI_INT, i, 3, MPI_COMM_WORLD);
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
