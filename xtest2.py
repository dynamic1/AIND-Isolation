from isolation import Board
from sample_players import RandomPlayer, GreedyPlayer, HumanPlayer

from visual_isolation import VisualBoard

from queue import Queue
from threading import Thread
from time import sleep

#import pygame

from xlogger import xlogger

from isolation import Board
from sample_players import RandomPlayer
from sample_players import null_score
from sample_players import open_move_score
from sample_players import improved_score
from game_agent import CustomPlayer
from game_agent import custom_score
# from game_agent import custom_score_x
from collections import namedtuple


import timeit

logger = xlogger()

Agent = namedtuple("Agent", ["player", "name"])

TIME_LIMIT = 150

HEURISTICS = [("Null", null_score),
              ("Open", open_move_score),
              ("Improved", improved_score)]
AB_ARGS = {"search_depth": 5, "method": 'alphabeta', "iterative": False}
MM_ARGS = {"search_depth": 3, "method": 'minimax', "iterative": False}
CUSTOM_ARGS = {"method": 'alphabeta', 'iterative': True}

# Create a collection of CPU agents using fixed-depth minimax or alpha beta
# search, or random selection.  The agent names encode the search method
# (MM=minimax, AB=alpha-beta) and the heuristic function (Null=null_score,
# Open=open_move_score, Improved=improved_score). For example, MM_Open is
# an agent using minimax search with the open moves heuristic.
mm_agents = [Agent(CustomPlayer(score_fn=h, **MM_ARGS),
                   "MM_" + name) for name, h in HEURISTICS]
ab_agents = [Agent(CustomPlayer(score_fn=h, **AB_ARGS),
                   "AB_" + name) for name, h in HEURISTICS]
random_agents = [Agent(RandomPlayer(), "Random")]

# ID_Improved agent is used for comparison to the performance of the
# submitted agent for calibration on the performance across different
# systems; i.e., the performance of the student agent is considered
# relative to the performance of the ID_Improved agent to account for
# faster or slower computers.
# test_agents = [Agent(CustomPlayer(score_fn=improved_score, **CUSTOM_ARGS), "ID_Improved"),
#               Agent(CustomPlayer(score_fn=custom_score, **CUSTOM_ARGS), "Student")]
#test_agents = [ Agent(CustomPlayer(score_fn=custom_score, **CUSTOM_ARGS), "Student"),
#                Agent(CustomPlayer(score_fn=improved_score, **CUSTOM_ARGS), "ID_Improved")]

my_agent = Agent(CustomPlayer(score_fn=custom_score, **CUSTOM_ARGS), "Student")
adversary = Agent(RandomPlayer(), "Random")

# game = Board(my_agent.player, adversary.player,7,7)

"""
game = Board(my_agent.player, adversary.player, 7, 7)
game.__board_state__= [[0,0,0,0,4,4,0],
                       [0,0,4,4,2,0,0],
                       [0,4,4,4,4,1,4],
                       [0,4,4,4,4,0,0],
                       [0,4,4,4,4,4,0],
                       [0,0,0,4,4,0,0],
                       [4,0,0,0,0,0,4]]

game.__last_player_move__ = {game.__player_1__: (2,5), game.__player_2__: (1,4)}
print(game.to_string())
TIME_LIMIT = 20

curr_time_millis = lambda: 1000 * timeit.default_timer()
move_start = curr_time_millis()
time_left = lambda: TIME_LIMIT - (curr_time_millis() - move_start)

move = my_agent.player.get_move(game, game.get_legal_moves(my_agent.player),time_left)

# winner, history, outcome = game.play()
# print("\nWinner: {}\nOutcome: {}".format(winner, outcome))
# print(game.to_string())
# print(history)
print(move)
"""

game = Board(my_agent.player, adversary.player, 7, 7)
print(game.to_string())

moves_by_turn = [[(2, 2), (6, 1)], [(3, 4), (5, 3)], [(4, 2), (4, 1)], [(2, 3), (6, 0)], [(4, 4), (5, 2)], [(3, 2), (3, 3)], [(2, 4), (1, 2)], [(4, 3), (0, 0)], [(3, 5), (2, 1)], [(1, 4), (0, 2)], [(2, 6), (1, 0)], [(4, 5), (3, 1)], [(6, 4), (5, 0)], [(5, 6), (6, 2)], [(-1, -1)]]
turn_count=0
TIME_LIMIT = 1000
for turn in moves_by_turn:

    turn_count = turn_count +1
    p1_move = turn[0]
    p2_move = turn[1] if len(turn) ==2 else None

    curr_time_millis = lambda: 1000 * timeit.default_timer()
    move_start = curr_time_millis()
    time_left = lambda: TIME_LIMIT - (curr_time_millis() - move_start)
    my_best_move = my_agent.player.get_move(game, game.get_legal_moves(my_agent.player),time_left)

    same_move = my_best_move == p1_move
    msg_string = 'OK' if same_move else 'NOT!'
    print(f"turn {turn_count} my best move: {my_best_move}, recorded move {p1_move} {msg_string}")
    game.apply_move(p1_move)
    if game.utility(my_agent.player) != 0:
        print(game.to_string())
        if game.is_loser(my_agent.player):
            print ("game over, I lose")
        else:
            print ("game over, I win")
        break

    game.apply_move(p2_move)
    print(game.to_string())
    if game.utility(my_agent.player) != 0:
        if game.is_loser(my_agent.player):
            print ("game over, I lose")
        else:
            print ("game over, I win")
        break



    """
again = True
while again:
    game = Board(my_agent.player, adversary.player,7,7)
    print(game.to_string())

    curr_time_millis = lambda: 1000 * timeit.default_timer()
    move_start = curr_time_millis()
    time_left = lambda : TIME_LIMIT - (curr_time_millis() - move_start)

    # move = my_agent.player.get_move(game, game.get_legal_moves(my_agent.player),time_left)

    winner, history, outcome = game.play()
    print("\nWinner: {}\nOutcome: {}".format(winner, outcome))
    print(game.to_string())
    print(history)
    again = (winner == my_agent.player)

"""