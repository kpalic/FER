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
    """
    Funkcija za dodavanje poteza na ploču.
    """
    obrnuti_poredak_redaka = reversed(range(ROWS))  # Obrnuti redoslijed redaka
    for row in obrnuti_poredak_redaka:
        if ploca[row][column] == EMPTY:  # Pronađi prvi prazni redak
            ploca[row][column] = player  # Dodaj potez
            return row, column
    return None, None

def evaluate_board_job(board):
    """
    Funkcija za procjenu stanja ploče.
    """
    def is_player_winner(player):
        """
        Funkcija za provjeru je li igrač pobjednik.
        """
        # Provjeri dijagonalne pobjede (dolje-lijevo do gore-desno)
        for row in range(3, ROWS):
            for column in range(COLUMNS - 3):
                if (board[row][column] == player and board[row - 1][column + 1] == player and 
                    board[row - 2][column + 2] == player and board[row - 3][column + 3] == player):
                    return True

        # Provjeri vertikalne pobjede
        for row in range(ROWS - 3):
            for column in range(COLUMNS):
                if (board[row][column] == player and board[row + 1][column] == player and 
                    board[row + 2][column] == player and board[row + 3][column] == player):
                    return True

        # Provjeri dijagonalne pobjede (gore-lijevo do dolje-desno)
        for row in range(ROWS - 3):
            for column in range(COLUMNS - 3):
                if (board[row][column] == player and board[row + 1][column + 1] == player and 
                    board[row + 2][column + 2] == player and board[row + 3][column + 3] == player):
                    return True
            
        # Provjeri horizontalne pobjede
        for row in range(ROWS):
            for column in range(COLUMNS - 3):
                if (board[row][column] == player and board[row][column + 1] == player and 
                    board[row][column + 2] == player and board[row][column + 3] == player):
                    return True

        return False
    
    if is_player_winner(PLAYER):  # Provjeri je li igrač pobjednik
        #logging.debug(f"Player win is detected {board}")
        return -1
    elif is_player_winner(AI):  # Provjeri je li AI pobjednik
        #logging.debug(f"AI win is detected {board}")
        return 1
    else:  # Nema pobjednika
        return 0

def print_board(ploca):
    """
    Funkcija za ispis trenutnog stanja ploče.
    """
    #logging.info(f"Trenutno stanje ploce:\n{ploca}")
    print(ploca)
    print()

def generiraj_poslove(board, broj_razina, current_path=[]):
    """
    Funkcija za rekurzivno generiranje svih poslova (poteza).
    """
    if not broj_razina:
        return [tuple(current_path)]
    
    jobs = []
    for column in range(COLUMNS):
        next_board = board.copy()  # Kopiraj trenutnu ploču
        player = AI if len(current_path) % 2 == 0 else PLAYER  # Odredi koji igrač je na potezu
        row, column = odigraj_potez(next_board, column, player)  # Odigraj potez
        if row is None and column is None:  # Ako potez nije validan, preskoči
            continue
        jobs += generiraj_poslove(next_board, broj_razina - 1, current_path + [column])  # Rekurzivno generiraj sljedeće poteze
    return jobs

def calculate_scores(current_result, broj_razina, is_ai_turn):
    """
    Funkcija za rekurzivno izračunavanje vrijednosti čvorova stabla.
    """
    if broj_razina == 0:
        return current_result

    parent_results = {}
    for path, score in current_result.items():
        parent_path = path[:-1]  # Ukloni posljednji element puta
        if parent_path not in parent_results:
            parent_results[parent_path] = []
        parent_results[parent_path].append(score)

    # Generiraj score za čvorove po pravilima
    for parent_path, child_scores in parent_results.items():
        if is_ai_turn:
            if -1 in child_scores:  # Ako je AI na potezu i postoji potez koji vodi u poraz
                #logging.debug(f"AI turn: Node {parent_path} has a child with -1")
                parent_results[parent_path] = -1
            else:
                avg_children_score = sum(child_scores) / len(child_scores)  # Izračunaj prosjek
                #logging.debug(f"AI turn: Node {parent_path} with child_scores {child_scores} average score is {avg_children_score}")
                parent_results[parent_path] = avg_children_score
        else:
            if 1 in child_scores:  # Ako je igrač na potezu i postoji potez koji vodi u pobjedu
                #logging.debug(f"Player turn: Node {parent_path} has a child with 1")
                parent_results[parent_path] = 1
            elif all(score == -1 for score in child_scores):  # Ako svi potezi vode u poraz
                #logging.debug(f"Player turn: All children of node {parent_path} have score -1")
                parent_results[parent_path] = -1
            else:
                #logging.debug(f"Player turn: child_scores {child_scores}")
                avg_children_score = sum(child_scores) / len(child_scores)  # Izračunaj prosjek
                #logging.debug(f"Player turn: Node {parent_path} average score is {avg_children_score}")
                parent_results[parent_path] = avg_children_score

    return calculate_scores(parent_results, broj_razina - 1, not is_ai_turn)

def worker():
    """
    Funkcija za radnički proces koji obrađuje zadatke.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    #logging.info(f"Worker {rank} starting")
    status = MPI.Status()
    comm.send(None, dest=0, tag=0)  # Inicijalna prijava

    while True:
        message = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)  # Primanje poruke
        #logging.debug(f"Worker {rank} dobio posao {message}")
        if status.Get_tag() == 3:  # Ako je tag 3, nema više poslova
            #logging.info(f"Worker {rank} primio poruku da nema vise poslova, proces zavrsava")
            break

        board, pathToTaskNode = message
        #logging.debug(f"Worker {rank} evaluating path {pathToTaskNode}")

        trenutni_player = AI  # AI uvijek započinje
        for i in range(len(pathToTaskNode)):
            odigraj_potez(board, pathToTaskNode[i], trenutni_player)  # Odigraj potez
            trenutni_player = PLAYER if trenutni_player == AI else AI  # Promjena igrača

        worker_results = {}
        worker_jobs = generiraj_poslove(board, DUBINA_PRETRAZIVANJA - DUBINA_ZADATAKA)
        #logging.info(f"Worker {rank} generated jobs: {worker_jobs}")

        if not worker_jobs:  # Ako nema više mogućih sljedećih poslova
            result = evaluate_board_job(board)
            comm.send((pathToTaskNode, result), dest=0, tag=1)
            #logging.debug(f"Worker {rank} salje masteru vrijednost za pathToTaskNode {pathToTaskNode} {result}")
            continue

        while worker_jobs:
            worker_path = worker_jobs.pop(0)  # Uzmi posao
            #logging.debug(f"Worker {rank} uzima worker_job {worker_path}")
            temp_board = board.copy()
            trenutni_worker_player = trenutni_player
            for i in range(len(worker_path)):
                odigraj_potez(temp_board, worker_path[i], trenutni_worker_player)  # Odigraj potez
                trenutni_worker_player = PLAYER if trenutni_worker_player == AI else AI  # Promjena igrača

            result = evaluate_board_job(temp_board)  # Procijeni ploču
            worker_results[pathToTaskNode + worker_path] = result
            #logging.debug(f"Worker {rank} completed worker_job {worker_path} with result {result}")
        
        is_ai_turn = (DUBINA_PRETRAZIVANJA % 2 == 0)  # Ako vrijedi, AI je na potezu u čvorovima neposredno iznad listova
        
        final_results = calculate_scores(worker_results, DUBINA_PRETRAZIVANJA - DUBINA_ZADATAKA, is_ai_turn)  # Vraća vrijednost čvora pathToTaskNode
        
        score_to_send = None
        path_to_send = None

        for path, score in final_results.items():  # Samo jedan entry će biti i to za pathToTaskNode
            score_to_send = score
            path_to_send = path

        comm.send((path_to_send, score_to_send), dest=0, tag=1)
        #logging.debug(f"Worker {rank} salje masteru worker_job i rezultat {path_to_send} {score_to_send}")

def master():
    """
    Funkcija za glavni proces koji upravlja igrom i raspodjeljuje poslove.
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    #logging.info(f"Master {rank} starting with {size} processes")
    board = np.zeros((ROWS, COLUMNS), dtype=int)  # Inicijalizacija ploče
    game_over = False
    workers_needing_job = []

    while not game_over:
        print_board(board)  # Ispis ploče
        column = int(input(f"Select a column (0-6): "))  # Unos poteza igrača
        odigraj_potez(board, column, PLAYER)  # Odigraj potez

        start_time = time.perf_counter()

        if evaluate_board_job(board) == -1:  # Provjera je li igrač pobjednik
            print_board(board)
            print("Player wins")
            game_over = True
            break

        # Generiranje poslova za proizvoljnu dubinu
        jobs = generiraj_poslove(board, DUBINA_ZADATAKA)
        num_of_jobs = len(jobs)
        #logging.info(f"Master {rank} generated {num_of_jobs} jobs: {jobs}")
        results = {}

        if num_of_jobs == 0:  # Ako nema više poslova, igra je neriješena
            #logging.info("Draw")
            game_over = True
            break

        status = MPI.Status()

        while len(results) < num_of_jobs:
            # Provjera dolaznih poruka
            while jobs and workers_needing_job:
                job = jobs.pop(0)  # Uzmi posao
                free_worker = workers_needing_job.pop(0)  # Uzmi slobodnog radnika
                #logging.debug(f"Sending job {job} to worker {free_worker}")
                comm.send((board.copy(), job), dest=free_worker, tag=1)  # Pošalji posao radniku
                
            if comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status):
                message = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)  # Primanje poruke
                worker_rank = status.Get_source()
                #logging.debug(f"Received message from worker {worker_rank}")

                if status.Get_tag() == 0:  # Prijava radnika
                    if jobs:
                        job = jobs.pop(0)
                        comm.send((board.copy(), job), dest=worker_rank, tag=1)  # Pošalji posao radniku
                        #logging.debug(f"Master sent to worker {worker_rank} a job {job}")
                    else:
                        workers_needing_job.append(worker_rank)  # Dodaj radnika na listu čekanja
                else:
                    #logging.debug(f"Worker {worker_rank} returned result: {message}")>
                    path, result = message
                    results[path] = result
                    if jobs:
                        job = jobs.pop(0)
                        comm.send((board.copy(), job), dest=worker_rank, tag=1)  # Pošalji posao radniku
                        #logging.debug(f"Master sent to worker {worker_rank} a job {job}")
                    else:
                        workers_needing_job.append(worker_rank)  # Dodaj radnika na listu čekanja

            elif jobs:  # Master također može obavljati posao
                job = jobs.pop(0)
                #logging.debug(f"Master evaluating path {job}")
                temp_board = board.copy()
                pathToTaskNode = tuple(job)

                trenutni_player = AI  # AI uvijek započinje
                for i in range(len(pathToTaskNode)):
                    odigraj_potez(temp_board, pathToTaskNode[i], trenutni_player)  # Odigraj potez
                    trenutni_player = PLAYER if trenutni_player == AI else AI  # Promjena igrača

                master_as_worker_results = {}
                master_as_worker_jobs = generiraj_poslove(board, DUBINA_PRETRAZIVANJA - DUBINA_ZADATAKA)
                #logging.info(f"Master {rank} generated jobs: {master_as_worker_jobs}")

                if not master_as_worker_jobs:  # Ako nema više mogućih sljedećih poslova
                    result = evaluate_board_job(board)
                    results[pathToTaskNode] = result
                    #logging.debug(f"Master {rank} je obavio posao pathToTaskNode {pathToTaskNode} {result}")
                    continue

                while master_as_worker_jobs:
                    master_as_worker_path = master_as_worker_jobs.pop(0)  # Uzmi posao
                    #logging.debug(f"Master {rank} uzima worker_job {master_as_worker_path}")
                    temp_temp_board = temp_board.copy()
                    trenutni_worker_player = trenutni_player
                    for i in range(len(master_as_worker_path)):
                        odigraj_potez(temp_temp_board, master_as_worker_path[i], trenutni_worker_player)  # Odigraj potez
                        trenutni_worker_player = PLAYER if trenutni_worker_player == AI else AI  # Promjena igrača

                    result = evaluate_board_job(temp_temp_board)  # Procijeni ploču
                    master_as_worker_results[pathToTaskNode + master_as_worker_path] = result
                    #logging.debug(f"Master {rank} completed worker_job {master_as_worker_path} with result {result}")
                
                is_ai_turn = (DUBINA_PRETRAZIVANJA % 2 == 0)  # Ako vrijedi, AI je na potezu u čvorovima neposredno iznad listova

                final_results = calculate_scores(master_as_worker_results, DUBINA_PRETRAZIVANJA - DUBINA_ZADATAKA, is_ai_turn)  # Vraća vrijednost čvora pathToTaskNode

                for pathToTaskNode, result in final_results.items():  # Samo jedan entry će biti i to za pathToTaskNode
                    results[pathToTaskNode] = result
                    #logging.debug(f"Master {rank} obavio posao {pathToTaskNode} {result}")

        is_ai_turn = (DUBINA_ZADATAKA % 2 == 0)  # Ako vrijedi, AI je na potezu u čvorovima neposredno iznad listova
        #logging.debug(f"Initial results: {results}")
        final_results = calculate_scores(results, DUBINA_ZADATAKA - 1, is_ai_turn)  # Vrijednosti čvorova na prvoj razini ispod korijena
        #logging.debug(f"Final results: {final_results}")

        best_score = -2
        best_column = None

        for column, score in final_results.items():
            if score > best_score:
                best_score = score
                best_column = column[0]

        if best_column is not None:
            row, column = odigraj_potez(board, best_column, AI)  # Odigraj najbolji potez za AI
            if evaluate_board_job(board) == 1:  # Provjeri je li AI pobjednik
                print_board(board)
                print("AI wins")
                game_over = True
                break
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Trajanje izvrsavanja: {duration:.4f} sekundi")

    # Pošalji svim radnicima poruku da nema više poslova
    for i in range(1, size):
        comm.send(None, dest=i, tag=3)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    if rank == 0:
        master()  # Pokreni master proces
    else:
        worker()  # Pokreni worker proces
