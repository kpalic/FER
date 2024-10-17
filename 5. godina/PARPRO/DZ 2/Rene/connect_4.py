from mpi4py import MPI
import numpy as np
import logging
import time

# naredba: mpiexec -n 8 -affinity -cores 1 python connect_4.py

ROWS = 6
COLUMNS = 7
EMPTY = 0
PLAYER = 1
AI = 2
DUBINA_ZADATAKA = 5
DUBINA_PRETRAZIVANJA = 6
#Mora vrijediti DUBINA_PRETRAZIVANJA >= DUBINA_ZADATAKA

# Postavljanje logiranja
logging.basicConfig(filename=f'connect_4_rank_{MPI.COMM_WORLD.Get_rank()}.log',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')

def odigraj_potez(ploca, column, player):
    obrnuti_poredak_redaka = reversed(range(ROWS))
    for row in obrnuti_poredak_redaka:
        if ploca[row][column] == EMPTY:
            ploca[row][column] = player
            return row, column
    return None, None

def evaluate_board_job(board):
    def is_player_winner(player):
        for row in range(3, ROWS):
            for column in range(COLUMNS - 3):
                if board[row][column] == player and board[row - 1][column + 1] == player and board[row - 2][column + 2] == player and board[row - 3][column + 3] == player:
                    return True

        for row in range(ROWS - 3):
            for column in range(COLUMNS):
                if board[row][column] == player and board[row + 1][column] == player and board[row + 2][column] == player and board[row + 3][column] == player:
                    return True

        for row in range(ROWS - 3):
            for column in range(COLUMNS - 3):
                if board[row][column] == player and board[row + 1][column + 1] == player and board[row + 2][column + 2] == player and board[row + 3][column + 3] == player:
                    return True
            
        for row in range(ROWS):
            for column in range(COLUMNS - 3):
                if board[row][column] == player and board[row][column + 1] == player and board[row][column + 2] == player and board[row][column + 3] == player:
                    return True

        return False
    
    if is_player_winner(PLAYER):
        #logging.debug(f"Player win is detected {board}")
        return -1
    elif is_player_winner(AI):
        #logging.debug(f"AI win is detected {board}")
        return 1
    else:
        return 0

def print_board(ploca):
    #logging.info(f"Trenutno stanje ploce:\n{ploca}")
    print(ploca)
    print()

def generiraj_poslove(board, broj_razina, current_path=[]): #rekurzivno generiranje svih poslova
    if not broj_razina:
        return [tuple(current_path)]
    
    jobs = []
    for column in range(COLUMNS):
        next_board = board.copy()
        player = AI if len(current_path) % 2 == 0 else PLAYER
        row, column = odigraj_potez(next_board, column, player)
        if row is None and column is None:
            continue
        jobs += generiraj_poslove(next_board, broj_razina - 1, current_path + [column])
    return jobs

def calculate_scores(current_result, broj_razina, is_ai_turn):
    if broj_razina == 0:
        return current_result

    parent_results = {}
    for path, score in current_result.items():
        parent_path = path[:-1]
        if parent_path not in parent_results:
            parent_results[parent_path] = []
        parent_results[parent_path].append(score)

    #generiranje scorea za cvorova po pravilima
    for parent_path, child_scores in parent_results.items():
        if is_ai_turn:
            if -1 in child_scores:
                #logging.debug(f"AI turn: Node {parent_path} has a child with -1")
                parent_results[parent_path] = -1
            else:
                avg_children_score = sum(child_scores) / len(child_scores)
                #logging.debug(f"AI turn: Node {parent_path} with child_scores {child_scores} average score is {avg_children_score}")
                parent_results[parent_path] = avg_children_score
        else:
            if 1 in child_scores:
                #logging.debug(f"Player turn: Node {parent_path} has a child with 1")
                parent_results[parent_path] = 1
            elif all(score == -1 for score in child_scores):
                #logging.debug(f"Player turn: All children of node {parent_path} have score -1")
                parent_results[parent_path] = -1
            else:
                #logging.debug(f"Player turn: child_scores {child_scores}")
                avg_children_score = sum(child_scores) / len(child_scores)
                #logging.debug(f"Player turn: Node {parent_path} average score is {avg_children_score}")
                parent_results[parent_path] = avg_children_score

    return calculate_scores(parent_results, broj_razina - 1, not is_ai_turn)

def worker():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    #logging.info(f"Worker {rank} starting")
    status = MPI.Status()
    comm.send(None, dest=0, tag=0) # inicijalna prijava

    while True:
        message = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        #logging.debug(f"Worker {rank} dobio posao {message}")
        if status.Get_tag() == 3:
            #logging.info(f"Worker {rank} primio poruku da nema vise poslova, proces zavrsava")
            break

        board, pathToTaskNode = message
        #logging.debug(f"Worker {rank} evaluating path {pathToTaskNode}")

        trenutni_player = AI  #AI je uvijek započinje
        for i in range(len(pathToTaskNode)):
            odigraj_potez(board, pathToTaskNode[i], trenutni_player)
            trenutni_player = PLAYER if trenutni_player == AI else AI  # Promjena igrača

        worker_results = {}
        worker_jobs = generiraj_poslove(board, DUBINA_PRETRAZIVANJA - DUBINA_ZADATAKA)
        #logging.info(f"Worker {rank} generated jobs: {worker_jobs}")

        if not worker_jobs: # nema vise mogucih slijednjih poslova ili DUBINA_PRETRAZIVANJA == DUBINA_ZADATAKA
            result = evaluate_board_job(board)
            comm.send((pathToTaskNode, result), dest=0, tag=1)
            #logging.debug(f"Worker {rank} salje masteru vrijednost za pathToTaskNode {pathToTaskNode} {result}")
            continue

        while worker_jobs:
            worker_path = worker_jobs.pop(0)
            #logging.debug(f"Worker {rank} uzima worker_job {worker_path}")
            temp_board = board.copy()
            trenutni_worker_player = trenutni_player
            for i in range(len(worker_path)):
                odigraj_potez(temp_board, worker_path[i], trenutni_worker_player)
                trenutni_worker_player = PLAYER if trenutni_worker_player == AI else AI  # Promjena igrača

            result = evaluate_board_job(temp_board)
            worker_results[pathToTaskNode + worker_path] = result
            #logging.debug(f"Worker {rank} completed worker_job {worker_path} with result {result}")
        
        is_ai_turn = (DUBINA_PRETRAZIVANJA % 2 == 0)  # ako to vrijedi AI je na potezu u cvorovima neposredno iznad listova
        
        final_results = calculate_scores(worker_results, DUBINA_PRETRAZIVANJA - DUBINA_ZADATAKA, is_ai_turn) # vraca vrijednost cvora pathToTaskNode
        
        score_to_send = None
        path_to_send = None

        for path, score in final_results.items(): # samo jedan entry ce bit i to za pathToTaskNode
            score_to_send = score
            path_to_send = path

        comm.send((path_to_send, score_to_send), dest=0, tag=1)
        #logging.debug(f"Worker {rank} salje masteru worker_job i rezultat {path_to_send} {score_to_send}")

def master():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    #logging.info(f"Master {rank} starting with {size} processes")
    board = np.zeros((ROWS, COLUMNS), dtype=int)
    game_over = False
    workers_needing_job = []

    while not game_over:
        print_board(board)
        column = int(input(f"Select a column (0-6): "))
        odigraj_potez(board, column, PLAYER)

        start_time = time.perf_counter()

        if evaluate_board_job(board) == -1:
            print_board(board)
            print("Player wins")
            game_over = True
            break

        # Generiranje poslova za proizvoljnu dubinu
        jobs = generiraj_poslove(board, DUBINA_ZADATAKA)
        num_of_jobs = len(jobs)
        #logging.info(f"Master {rank} generated {num_of_jobs} jobs: {jobs}")
        results = {}

        if num_of_jobs == 0:
            #logging.info("Draw")
            game_over = True
            break

        status = MPI.Status()

        while len(results) < num_of_jobs:
            # Provjera dolaznih poruka
            while jobs and workers_needing_job:
                job = jobs.pop(0)
                free_worker = workers_needing_job.pop(0)
                #logging.debug(f"Sending job {job} to worker {free_worker}")
                comm.send((board.copy(), job), dest=free_worker, tag=1)
                
            if comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status):
                message = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                worker_rank = status.Get_source()
                #logging.debug(f"Received message from worker {worker_rank}")

                if status.Get_tag() == 0:  # Prijava radnika
                    if jobs:
                        job = jobs.pop(0)
                        comm.send((board.copy(), job), dest=worker_rank, tag=1)
                        #logging.debug(f"Master sent to worker {worker_rank} a job {job}")
                    else:
                        workers_needing_job.append(worker_rank)
                else:
                    #logging.debug(f"Worker {worker_rank} returned result: {message}")>
                    path, result = message
                    results[path] = result
                    if jobs:
                        job = jobs.pop(0)
                        comm.send((board.copy(), job), dest=worker_rank, tag=1)
                        #logging.debug(f"Master sent to worker {worker_rank} a job {job}")
                    else:
                        workers_needing_job.append(worker_rank)

            elif jobs: # master je implementiran da radi isto kao i workeri
                job = jobs.pop(0)
                #logging.debug(f"Master evaluating path {job}")
                temp_board = board.copy()
                pathToTaskNode = tuple(job)

                trenutni_player = AI  #AI je uvijek započinje
                for i in range(len(pathToTaskNode)):
                    odigraj_potez(temp_board, pathToTaskNode[i], trenutni_player)
                    trenutni_player = PLAYER if trenutni_player == AI else AI  # Promjena igrača

                master_as_worker_results = {}
                master_as_worker_jobs = generiraj_poslove(board, DUBINA_PRETRAZIVANJA - DUBINA_ZADATAKA)
                #logging.info(f"Master {rank} generated jobs: {master_as_worker_jobs}")

                if not master_as_worker_jobs: # nema vise mogucih slijednjih poslova ili DUBINA_PRETRAZIVANJA == DUBINA_ZADATAKA
                    result = evaluate_board_job(board)
                    results[pathToTaskNode] = result
                    #logging.debug(f"Master {rank} je obavio posao pathToTaskNode {pathToTaskNode} {result}")
                    continue

                while master_as_worker_jobs:
                    master_as_worker_path = master_as_worker_jobs.pop(0)
                    #logging.debug(f"Master {rank} uzima worker_job {master_as_worker_path}")
                    temp_temp_board = temp_board.copy()
                    trenutni_worker_player = trenutni_player
                    for i in range(len(master_as_worker_path)):
                        odigraj_potez(temp_temp_board, master_as_worker_path[i], trenutni_worker_player)
                        trenutni_worker_player = PLAYER if trenutni_worker_player == AI else AI  # Promjena igrača

                    result = evaluate_board_job(temp_temp_board)
                    master_as_worker_results[pathToTaskNode + master_as_worker_path] = result
                    #logging.debug(f"Master {rank} completed worker_job {master_as_worker_path} with result {result}")
                
                is_ai_turn = (DUBINA_PRETRAZIVANJA % 2 == 0)  # ako to vrijedi AI je na potezu u cvorovima neposredno iznad listova

                final_results = calculate_scores(master_as_worker_results, DUBINA_PRETRAZIVANJA - DUBINA_ZADATAKA, is_ai_turn) # vraca vrijednost cvora pathToTaskNode

                for pathToTaskNode, result in final_results.items(): # samo jedan entry ce bit i to za pathToTaskNode
                    results[pathToTaskNode] = result
                    #logging.debug(f"Master {rank} obavio posao {pathToTaskNode} {result}")

        is_ai_turn = (DUBINA_ZADATAKA % 2 == 0)  # ako to vrijedi AI je na potezu u cvorovima neposredno iznad listova
        #logging.debug(f"Initial results: {results}")
        final_results = calculate_scores(results, DUBINA_ZADATAKA - 1, is_ai_turn) # vrijednosti cvorova na prvoj razini ispod korijena
        #logging.debug(f"Final results: {final_results}")

        best_score = -2
        best_column = None

        for column, score in final_results.items():
            if score > best_score:
                best_score = score
                best_column = column[0]

        if best_column is not None:
            row, column = odigraj_potez(board, best_column, AI)
            if evaluate_board_job(board) == 1:
                print_board(board)
                print("AI wins")
                game_over = True
                break
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Trajanje izvrsavanja: {duration:.4f} sekundi")

    # Posalji svim workerima poruku da nema vise poslova
    for i in range(1, size):
        comm.send(None, dest=i, tag=3)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    if rank == 0:
        master()
    else:
        worker()
