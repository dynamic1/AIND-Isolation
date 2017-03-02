from isolation import Board
from sample_players import RandomPlayer, GreedyPlayer, HumanPlayer

from visual_isolation import VisualBoard

from queue import Queue
from threading import Thread
from time import sleep

import pygame

# will use q to exchenge messages between threads
q = Queue()

# this will be the thread doing the solving
def worker():

    # create an isolation board (by default 7x7)
    player1 = RandomPlayer()
    player2 = GreedyPlayer()
    game = Board(player1, player2)
    game.apply_move((2, 3))
    q.put(game)
    sleep(2)
    game.apply_move((0, 5))
    q.put(game)
    sleep(2)

    # play the remainder of the game automatically -- outcome can be "illegal
    # move" or "timeout"; it should _always_ be "illegal move" in this example
    winner, history, outcome = game.play(q)
    print("\nWinner: {}\nOutcome: {}".format(winner, outcome))


t = Thread(target=worker)
t.daemon = True
t.start()

visualization = VisualBoard(7,7,100,100)
visualization.main_loop(q)

