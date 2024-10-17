#include <iostream>
#include <string>
#include <utility>
#include <chrono>
#include <stack>
#include <vector>
#include <map>
#include <mpi.h>

#define BOARD_HEIGHT 6
#define BOARD_WIDTH 7
#define PLAYER 1
#define AI 2

#define TASK_DEPTH 5
#define SEARCH_DEPTH 6

#define JOB_ASSIGNED_BOARD_TAG 0
#define JOB_ASSIGNED_FULL_PATH_SIZE_TAG 1
#define JOB_ASSIGNED_PATH_SIZE_TAG 2
#define JOB_ASSIGNED_PATH_TAG 3

#define JOB_FINISHED_PATH_TAG 4
#define JOB_FINISHED_RESULT_TAG 5 

#define WORKERS_REPORTING_FOR_DUTY_TAG 6
#define GAME_FINISHED_TAG 7


using namespace std;

void printBoard(const vector<vector<char>>& board) {
    for (int i = 0; i < BOARD_HEIGHT; ++i) {
        for (int j = 0; j < BOARD_WIDTH; ++j) {
            cout << "| " << board[i][j] << " ";
        }
        cout << "|\n";
    }
    for (int j = 0; j < BOARD_WIDTH; ++j) {
        cout << "  " << j << " ";
    }
    cout << "\n";
}

bool validateMove(const vector<vector<char>>& board, int move) {
    if (move < 0 || move >= BOARD_WIDTH) {
        return false;
    }
    if (board[0][move] != ' ') {
        return false;
    }
    return true;
}

pair<int, int> makeMove(int player, vector<vector<char>>& board, int colNumber) {
    char playerMove = player == PLAYER ? 'X' : 'O';
    int rowNumber = -1;

    for (int i = BOARD_HEIGHT - 1; i >= 0; --i) {
        if (board[i][colNumber] == ' ') {
            board[i][colNumber] = playerMove;
            rowNumber = i;
            break;
        }
    }

    return make_pair(rowNumber, colNumber);
}

int checkWin(int colNumber, const vector<vector<char>>& board, int player) {
    int rowNumber = BOARD_HEIGHT;
    for (int i = 0; i < BOARD_HEIGHT; ++i) {
        if (board[i][colNumber] != ' ') {
            rowNumber = i;
            break;
        }
    }

    char playerMove = player == PLAYER ? 'X' : 'O';

    // Check horizontal
    int count = 0;
    for (int i = 0; i < BOARD_WIDTH; ++i) {
        if (board[rowNumber][i] == playerMove) {
            count++;
            if (count == 4) {
                return (player == AI) ? 1 : -1;
            }
        } else {
            count = 0;
        }
    }

    // Check vertical
    count = 0;
    for (int i = 0; i < BOARD_HEIGHT; ++i) {
        if (board[i][colNumber] == playerMove) {
            count++;
            if (count == 4) {
                return (player == AI) ? 1 : -1;
            }
        } else {
            count = 0;
        }
    }

    // Check left diagonal containing the last move
    count = 0;
    for (int i = rowNumber - 3, j = colNumber - 3; i <= rowNumber + 3 && j <= colNumber + 3; i++, j++) {
        if (i >= 0 && i < BOARD_HEIGHT && j >= 0 && j < BOARD_WIDTH) {
            if (board[i][j] == playerMove) {
                count++;
                if (count == 4) {
                    return (player == AI) ? 1 : -1;
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
            if (board[i][j] == playerMove) {
                count++;
                if (count == 4) {
                    return (player == AI) ? 1 : -1;
                }
            } else {
                count = 0;
            }
        }
    }
    return 0;
}

void getAvailableJobs(const vector<vector<char>>& board, int depth, vector<vector<int>>& jobs, vector<int> currentPath = {}) {
    if (depth == 0) {
        jobs.push_back(currentPath);
        return;
    }


    for (int column = 0; column < BOARD_WIDTH; ++column) {
        vector<vector<char>> newBoard = board;
        int player = currentPath.size() % 2 == 0 ? PLAYER : AI;
        pair<int, int> move = makeMove(player, newBoard, column);
        if (move.first == -1) {
            continue;
        }
        currentPath.push_back(column);
        getAvailableJobs(newBoard, depth - 1, jobs, currentPath);
        currentPath.pop_back();
    }
}

void sendJob(vector<vector<char>> board, vector<vector<int>> job, int worker) {
    vector<char> flattenBoard = {};
    for (int i = 0; i < BOARD_HEIGHT; ++i) {
        for (int j = 0; j < BOARD_WIDTH; ++j) {
            flattenBoard.push_back(board[i][j]);
        }
    }

    MPI_Send(flattenBoard.data(), BOARD_HEIGHT * BOARD_WIDTH, MPI_CHAR, worker, JOB_ASSIGNED_BOARD_TAG, MPI_COMM_WORLD);

    int jobSize = job.size();
    MPI_Send(&jobSize, 1, MPI_INT, worker, JOB_ASSIGNED_FULL_PATH_SIZE_TAG, MPI_COMM_WORLD);
    for (int i = 0; i < job.size(); i++) {
        int pathSize = job[i].size();
        MPI_Send(&pathSize, 1, MPI_INT, worker, JOB_ASSIGNED_PATH_SIZE_TAG, MPI_COMM_WORLD);
        MPI_Send(job[i].data(), pathSize, MPI_INT, worker, JOB_ASSIGNED_PATH_TAG, MPI_COMM_WORLD);
    }
}

void receiveJob(vector<vector<char>>* board, vector<vector<int>>* job) {
    MPI_Status status;
    char flattenBoard[BOARD_HEIGHT * BOARD_WIDTH];
    
    MPI_Recv(flattenBoard, BOARD_HEIGHT * BOARD_WIDTH, MPI_CHAR, 0, JOB_ASSIGNED_BOARD_TAG, MPI_COMM_WORLD, &status);
    for (int i = 0; i < BOARD_HEIGHT; ++i) {
        for (int j = 0; j < BOARD_WIDTH; ++j) {
            (*board)[i][j] = flattenBoard[i * BOARD_WIDTH + j];
        }
    }

    int jobSize;
    MPI_Recv(&jobSize, 1, MPI_INT, 0, JOB_ASSIGNED_FULL_PATH_SIZE_TAG, MPI_COMM_WORLD, &status);
    for (int i = 0; i < jobSize; i++) {
        int pathSize;
        MPI_Recv(&pathSize, 1, MPI_INT, 0, JOB_ASSIGNED_PATH_SIZE_TAG, MPI_COMM_WORLD, &status);
        vector<int> path(pathSize);
        MPI_Recv(path.data(), pathSize, MPI_INT, 0, JOB_ASSIGNED_PATH_TAG, MPI_COMM_WORLD, &status);
        job->push_back(path);
    }

}

void receiveResultFromWorker(double* result, int* path, int worker) {
    MPI_Status status;
    // cout << "Master received result from worker " << worker << endl;
    MPI_Recv(result, 1, MPI_DOUBLE, worker, JOB_FINISHED_RESULT_TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(path, 1, MPI_INT, worker, JOB_FINISHED_PATH_TAG, MPI_COMM_WORLD, &status);
}

void sendResultToMaster(int path, double result) {
    MPI_Send(&result, 1, MPI_DOUBLE, 0, JOB_FINISHED_RESULT_TAG, MPI_COMM_WORLD);
    MPI_Send(&path, 1, MPI_INT, 0, JOB_FINISHED_PATH_TAG, MPI_COMM_WORLD);
}

void worker() {
    MPI_Status status;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cout << "Worker " << rank << " started" << endl;
    // worker reporting for duty
    MPI_Send(nullptr, 0, MPI_INT, 0, WORKERS_REPORTING_FOR_DUTY_TAG, MPI_COMM_WORLD);
    cout << "Worker " << rank << " reported for duty" << endl;

    while (true) {
        int flag = 0;
        MPI_Iprobe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        if (status.MPI_TAG == GAME_FINISHED_TAG) {
            MPI_Recv(nullptr, 0, MPI_INT, 0, GAME_FINISHED_TAG, MPI_COMM_WORLD, &status);
            return;
        }
        else {
            vector<vector<char>> board(BOARD_HEIGHT, vector<char>(BOARD_WIDTH, ' '));
            vector<vector<int>> jobs;
            receiveJob(&board, &jobs);
            pair<int, double> result;

            for (int i = 0; i < jobs.size(); i++) {
                vector<vector<char>> newBoard = board;
                int playerMove = AI;
                bool foundWinningMove = false;
                int depthWinningMove = -1;
                int win = 0;
                for (int j = 0; j < jobs[i].size(); j++) {
                    playerMove = (j % 2 == 0) ? AI : PLAYER;
                    makeMove(playerMove, newBoard, jobs[i][j]);
                    win = checkWin(jobs[i][j], newBoard, playerMove);

                    if (win != 0) {
                        depthWinningMove = j;
                        break;
                    }
                }
                double tempResult = -2;
                if (depthWinningMove != -1) {
                    // add small random value to avoid division by zero
                    double random = (rand() % 100) / 10000.0;
                    tempResult = ((win * 1.0) / (depthWinningMove+1)) + random;
                }

                if (tempResult > result.second) {
                    result.first = jobs[i][0];
                    result.second = tempResult;
                }
            }
            sendResultToMaster(result.first, result.second);
        }
    }
}

void master() {
    vector<vector<char>> board(BOARD_HEIGHT, vector<char>(BOARD_WIDTH, ' '));
    bool gameOver = false;
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    cout << "Master starting with " << size << " processes\n";
    int comm = MPI_COMM_WORLD;
    vector<int> unemployedWorkers;

    while (!gameOver) {
        int colNumber;
        printBoard(board);
        cout << "Enter a column number: ";
        cin >> colNumber;

        while (!validateMove(board, colNumber)) {
            cout << "Invalid move. Enter a column number: ";
            cin >> colNumber;
        }

        makeMove(PLAYER, board, colNumber);
        printBoard(board);
        if (checkWin(colNumber, board, PLAYER) == -1) {
            cout << "Player wins!" << endl;
            gameOver = true;
            break;
        }

        vector<vector<int>> jobs;
        getAvailableJobs(board, TASK_DEPTH, jobs);

        // cout << "Master generated " << jobs.size() << " jobs" << endl;
        map<int, double> results;

        if (jobs.size() == 0) {
            cout << "Draw!" << endl;
            gameOver = true;
            break;
        }

        if (!gameOver) {
            MPI_Status status;
            while (jobs.size() > 0 && !gameOver) {
                bool MadeMoveAI = false;
                // check for messages
                while (jobs.size() > 0 && unemployedWorkers.size() > 0) {
                    int worker = unemployedWorkers.back();
                    unemployedWorkers.pop_back();
                    // send up to 100 jobs to workers
                    vector<vector<int>> jobsToSend;
                    for (int i = 0; i < 100 && jobs.size() > 0; i++) {
                        jobsToSend.push_back(jobs.back());
                        jobs.pop_back();
                    }
                    sendJob(board, jobsToSend, worker);
                }
                int flag = 0;
                MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, &status);
                if (flag) {
                    int path;
                    double result;
                    if (status.MPI_TAG == WORKERS_REPORTING_FOR_DUTY_TAG) { // Worker is reporting for duty
                        cout << "Worker " << status.MPI_SOURCE << " is assigned for duty\n";
                        MPI_Recv(nullptr, 0, MPI_INT, status.MPI_SOURCE, WORKERS_REPORTING_FOR_DUTY_TAG, MPI_COMM_WORLD, &status);
                        if (jobs.size() > 0) {
                            // send up to 100 jobs to workers
                            vector<vector<int>> jobsToSend;
                            for (int i = 0; i < 100 && jobs.size() > 0; i++) {
                                jobsToSend.push_back(jobs.back());
                                jobs.pop_back();
                            }
                            sendJob(board, jobsToSend, status.MPI_SOURCE);
                        } 
                        else {
                            unemployedWorkers.push_back(status.MPI_SOURCE);
                        }
                    }
                    else if (status.MPI_TAG == JOB_FINISHED_RESULT_TAG || status.MPI_TAG == JOB_FINISHED_PATH_TAG) { // Worker has finished the job
                        // print status
                        receiveResultFromWorker(&result, &path, status.MPI_SOURCE);
                        double bestResult;
                        auto it = results.find(path);
                        if (it != results.end()) {
                            bestResult = it->second;
                            if (result > bestResult) {
                                results[path] = result;
                            }
                        } else {
                            results.insert({path, result});
                        }
                        if (jobs.size() > 0) {
                            vector<vector<int>> jobsToSend;
                            for (int i = 0; i < 100 && jobs.size() > 0; i++) {
                                jobsToSend.push_back(jobs.back());
                                jobs.pop_back();
                            }
                            sendJob(board, jobsToSend, status.MPI_SOURCE);
                        } 
                        else {
                            unemployedWorkers.push_back(status.MPI_SOURCE);
                        }
                    }
                    flag = 0;
                    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, &status);
                }
            }

            if (!gameOver) {
                int bestPath = -1;
                double bestResult = -1;

                for (auto it = results.begin(); it != results.end(); ++it) {
                    if (it->second > bestResult) {
                        bestResult = it->second;
                        bestPath = it->first;
                        // cout << "Best path: " << bestPath << " with result: " << bestResult << endl;
                    }
                }
                // cout << "1" << endl;
                makeMove(AI, board, bestPath);
                // cout << "2" << endl;
                printBoard(board);
                // cout << "3" << endl;
                if (checkWin(bestPath, board, AI) == 1) {
                    cout << "AI wins!" << endl;
                    gameOver = true;
                    break;
                }
            }
        }

        
    }
    cout << "Game over" << endl;
    // send message to all workers that the game is over
    for (int i = 1; i < size; i++) {
        cout << "Thoughts and prayers for worker " << i << endl;
        MPI_Send(nullptr, 0, MPI_INT, i, GAME_FINISHED_TAG, MPI_COMM_WORLD);
    }
}

int main(int argc, char* argv[]) {
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