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

TIME_LIMIT = 10000

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

game = Board(my_agent.player, adversary.player,7,7)

print(' ', end='')
for x in range(7):
    print("%4d" % x, end='')
print()
for x in range(7):
    print(x,' ', end='')
    for y in range(7):
        current_game = game.forecast_move((x,y))
        # print(x,y, my_agent.player.score(current_game, my_agent.player), end=' ')
        print(my_agent.player.score(current_game, my_agent.player), end=' ')
    print()
"""
game.__board_state__= [[4,0,0,0,4,0,0],
                       [0,0,4,0,4,4,4],
                       [0,4,4,4,4,1,0],
                       [0,4,4,4,4,4,0],
                       [4,4,4,4,0,0,4],
                       [4,0,4,4,4,0,0],
                       [4,0,4,2,0,0,0]]
game.__last_player_move__ = {game.__player_1__: (2,5), game.__player_2__: (6,3)}
"""
print(game.to_string())

curr_time_millis = lambda: 1000 * timeit.default_timer()
move_start = curr_time_millis()
time_left = lambda : TIME_LIMIT - (curr_time_millis() - move_start)

move = my_agent.player.get_move(game, game.get_legal_moves(my_agent.player),time_left)
#print(move)

# logger.set_context("context1")
# logger.debug("message 1")
# logger.push_context(" context_2")
# logger.debug("message_2")
# logger.pop_context()
# logger.debug("message 3")