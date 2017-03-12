"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import math
import sys


"""
xalex
use logging for debug
"""
import logging
import logging.config

from xlogger import xlogger
from pprint import pprint


logging.config.fileConfig('logging.conf')

# create logger
#logger = logging.getLogger('simpleExample')
logger = xlogger()

## expale useage of logger
# logger.debug('debug message')
# logger.info('info message')
# logger.warning('warn message')
# logger.error('error message')
# logger.critical('critical message')



class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


#open move
def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return float(len( game.get_legal_moves(player) ) )

def custom_score_x(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    (x,y) = game.get_player_location(player)
    x = abs(game.width/2 - x) + abs(game.width/2 - y)

    return float(len( game.get_legal_moves(player) ) +x)

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def my_function_name(self):
        """
        :return: name of caller
        """

        """
        obj = sys._getframe(1).f_code
        for attr in dir(obj):
            print(            "obj.%s = %s\n" % (attr, getattr(obj, attr)))

        exit()
        """
        return sys._getframe(1).f_code.co_name

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
        # print()
        logger.push_context(f"get_move")
        logger.print(game.to_string())
        logger.debug(f'method={self.method} ID={self.iterative} ')
        logger.debug(f'START, time_lef={time_left()}, my pos is {game.get_player_location(game.active_player)}')

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        best_score = -math.inf
        best_move = (-1,-1)
        current_depth = 0 if self.iterative else self.search_depth
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            forecast_couter = 0

            while True:
                logger.push_context(f"get_move_{current_depth}")
                logger.debug(f'depth={current_depth}, legal_moves={legal_moves}')
                if len(legal_moves) <1:
                    logger.debug(f'depth={current_depth}, no legal moves, wil return (-1,-1)')
                    return best_move

                for move_idx, possible_move in enumerate(legal_moves):
                    possible_game = game.forecast_move(possible_move)
                    logger.debug(f'forecasting move {possible_move}')
                    logger.print(possible_game.to_string())
                    forecast_couter = forecast_couter + 1
                    # possible_game = game.forecast_move(possible_move)
                    logger.debug(f'time_left={round(time_left())}, evaluating move number {move_idx+1}, move={possible_move} with method {self.method}, depth={current_depth}')
                    # possible_score, recommended_move = self.minimax(possible_game,1,True)
                    if self.method == 'minimax':
                        possible_score, recommended_move = self.minimax(possible_game,current_depth, False )
                    else:
                        possible_score, recommended_move = self.alphabeta(possible_game, current_depth, -math.inf, math.inf, False)
                    logger.debug(
                        f'time_left={round(time_left())}, results for move number {move_idx}, move={possible_move} with method {self.method}, depth={current_depth}: {possible_move}-> {possible_score}')
                    if possible_score > best_score:
                        best_move = possible_move
                        best_score = possible_score

                if not self.iterative:
                    logger.warning(f"non iterative, finish with best move {best_move}")
                    logger.pop_context()
                    return best_move

                logger.debug(f"completed current_depth={current_depth}, best move {best_move}")
                logger.pop_context()
                current_depth = current_depth+1

        except Timeout:
            # Handle any actions required at timeout, if necessary
            logger.set_context("get_move")
            logger.debug(f"Aproaching timeout, will return best move so far {best_move}")
            return best_move

    def minimax_old(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()


        my_type = "MAX" if maximizing_player else "MIN"
        logger.debug(f'f={self.my_function_name()}: DEPTH={depth} MINIMAX start node type={my_type} depth={depth}\n{game.to_string()}' )


        best_move=(-1,-1)
        multiplier = 1 if maximizing_player else -1
        best_score = (- math.inf)

        my_player = game.active_player

        legal_moves = game.get_legal_moves(my_player)

        #check weather game would be over at this point
        utility =  game.utility(my_player)
        if utility != 0:
            return utility,game.get_player_location(my_player)

        if depth ==0 or len(legal_moves)==0:
            current_score = self.score(game,game.active_player)# if maximizing_player else possible_game.active_player)
            return (current_score, game.get_player_location(my_player))
            # logger.debug(f'move: {possible_move}: current_score={current_score}')

        moves_couter = 0
        forecast_couter = 0
        for possible_move in legal_moves:
            forecast_couter = forecast_couter + 1
            possible_game = game.forecast_move(possible_move)
            #logger.debug(f'DEPTH={depth} MIMIMAX doing forecast counter={forecast_couter}, move={possible_move}, new_counter={possible_game.counts[1]}')
            moves_couter = moves_couter+1
            if depth <= 1:
                # current_score = self.score(possible_game, possible_game.active_player if maximizing_player else possible_game.inactive_player)
                current_score = self.score(possible_game, possible_game.inactive_player if maximizing_player else possible_game.active_player)
                #logger.debug(f'move: {possible_move}: current_score={current_score}')

            else:
                logger.debug(f'f={self.my_function_name()}: DEPTH={depth} MINIMAX calling recursive, depth = {depth-1}, maximizing={not maximizing_player} to evaluate possible_move {possible_move}')
                current_score, optimal_move = self.minimax(possible_game, depth - 1, not maximizing_player)
                logger.debug(f'f={self.my_function_name()}: DEPTH={depth} move: {possible_move}: current_score={current_score}')
                pass

            if current_score* multiplier > best_score:
                logger.debug(
                    f'f={self.my_function_name()}: DEPTH={depth} choice {moves_couter}: move: {possible_move}: current_score={current_score}, multiplier={multiplier} ? best_score so far: {best_score} -> new best')
                best_score = current_score
                best_move = possible_move

            else:
                logger.debug(
                    f'f={self.my_function_name()}: DEPTH={depth} choice {moves_couter}: move: {possible_move}: current_score={current_score}, multiplier={multiplier} ? best_score so far: {best_score} -> ignore')

        logger.debug(f'f={self.my_function_name()}: DEPTH={depth} minimax END: best move {best_move}: score={best_score}')
        return (best_score, best_move)
        # return (0,(1,5))


    def minimax(self, game, depth, maximizing_player=True):
        """Implement minimax search as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # depth must be greater than or equal to 1
        assert depth>=0, "minimax depth must be at least 0"

        my_type = "MAX" if maximizing_player else "MIN"
        logger.push_context(f"minimax_{depth} {my_type} pos={game.get_player_location(game.active_player)}")
        logger.debug("start")
        # logger.debug(f'f={self.my_function_name()}: DEPTH={depth} {my_type} my pos is {game.get_player_location(game.active_player)}' )
        logger.print( game.to_string())

        best_move=(-1,-1)
        multiplier = 1 if maximizing_player else -1
        best_score = (- math.inf)

        my_player = game.active_player
        other_player = game.inactive_player

        # check weather game would be over at this point
        """
        utility = game.utility(other_player)
        logger.debug(f"utility = {utility}")
        if utility != 0:
            logger.debug(f'reached max depth: terminal move, utility={utility}')
            logger.pop_context()
            return utility, game.get_player_location(my_player)
        """

        current_score = self.score(game, other_player)
        current_move = game.get_player_location(game.inactive_player)

        # if depth is 0, will evaluate current branch using just the self.score function, no further recursion
        if depth ==0:
            logger.debug(f'reached max depth: {current_move} -> {current_score}')
            logger.pop_context()
            return (current_score, current_move)

        # will check for legal moves and recurse to evaluate each legal move
        legal_moves = game.get_legal_moves(my_player)

        # if there are no available moves
        if len(legal_moves) == 0:
            if (game.is_loser(my_player)):
                logger.debug(f'should not have reached this point: i lose')
                logger.pop_context()
                return (-math.inf, current_move)

            logger.error(f'should not have reached this point: do i win?')
            logger.pop_context()
            return (-math.inf, (-1, -1))



            # if there are no available moves
        if len(legal_moves)==0:
            logger.error(f'should not have reached this point:')
            # logger.debug(f'f={self.my_function_name()}: evaluating leaf: {current_move} -> {current_score}')
            logger.pop_context()
            return (-math.inf, (-1,-1))

        moves_couter = 0
        forecast_couter = 0
        logger.debug(f"there are {len(legal_moves)} possible moves: {legal_moves}")
        for possible_move in legal_moves:

            forecast_couter = forecast_couter + 1
            possible_game = game.forecast_move(possible_move)
            #logger.debug(f'DEPTH={depth} MIMIMAX doing forecast counter={forecast_couter}, move={possible_move}, new_counter={possible_game.counts[1]}')
            moves_couter = moves_couter+1

            logger.debug(f'calling recursive, depth = {depth-1}, maximizing={not maximizing_player} to evaluate possible_move {possible_move}')
            current_score, optimal_move = self.minimax(possible_game, depth - 1, not maximizing_player)
            logger.debug(f'move: {possible_move}: current_score={current_score}')

            if current_score* multiplier > best_score:
                logger.debug(
                    f'choice {moves_couter}: move: {possible_move}: current_score={current_score}, multiplier={multiplier} ? best_score so far: {best_score} -> new best')
                best_score = current_score
                best_move = possible_move

            else:
                logger.debug(
                    f'choice {moves_couter}: move: {possible_move}: current_score={current_score}, multiplier={multiplier} ? best_score so far: {best_score} -> ignore')


        logger.debug(f'END: best move {best_move}: score={best_score}')
        logger.pop_context()
        return (best_score, best_move)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # depth must be greater than or equal to 1
        assert depth>=0, "alphabeta depth must be at least 0"

        my_type = "MAX" if maximizing_player else "MIN"
        last_move = game.__last_player_move__[game.inactive_player]
        logger.push_context(f"alphabeta_{depth} {my_type} {last_move} pos={game.get_player_location(game.active_player)}")
        logger.debug("start")
        # logger.debug(f'f={self.my_function_name()}: DEPTH={depth} {my_type} my pos is {game.get_player_location(game.active_player)}' )
        logger.print( game.to_string())

        best_move=(-1,-1)
        multiplier = 1 if maximizing_player else -1
        best_score = (- math.inf)

        my_player = game.active_player
        other_player = game.inactive_player

        # check weather game would be over at this point
        """
        utility = game.utility(other_player)
        logger.debug(f"utility = {utility}")
        if utility != 0:
            logger.debug(f'reached max depth: terminal move, utility={utility}')
            logger.pop_context()
            return utility, game.get_player_location(my_player)
        """

        current_score = self.score(game, other_player)
        current_move = game.get_player_location(game.inactive_player)
        # if depth is 0, will evaluate current branch using just the self.score function, no further recursion
        if depth ==0:
            logger.debug(f'reached max depth: {current_move} -> {current_score}')
            logger.pop_context()
            return (current_score, current_move)

        # assert depth > 0
        # will check for legal moves and recurse to evaluate each legal move
        legal_moves = game.get_legal_moves(my_player)

        # if there are no available moves
        if len(legal_moves)==0:
            if(game.is_loser(my_player)):
                logger.debug(f'should not have reached this point: i lose')
                return (-math.inf,current_move)
            else:
                logger.error(f'should not have reached this point: do i win?')
            # logger.debug(f'f={self.my_function_name()}: evaluating leaf: {current_move} -> {current_score}')
            logger.pop_context()
            return (-math.inf, (-1,-1))

        moves_couter = 0
        forecast_couter = 0
        logger.debug(f"possible moves: {legal_moves}")
        for possible_move in legal_moves:

            forecast_couter = forecast_couter + 1
            possible_game = game.forecast_move(possible_move)
            #logger.debug(f'DEPTH={depth} MIMIMAX doing forecast counter={forecast_couter}, move={possible_move}, new_counter={possible_game.counts[1]}')
            moves_couter = moves_couter+1

            logger.debug(f'calling recursive, depth = {depth-1}, maximizing={not maximizing_player} to evaluate possible_move {possible_move}')
            current_score, optimal_move = self.alphabeta(possible_game, depth - 1, alpha, beta, not maximizing_player)
            logger.debug(f'move: {possible_move}: current_score={current_score}')

            if current_score* multiplier > best_score:
                logger.debug(
                    f'choice {moves_couter}: move: {possible_move}: current_score={current_score}, multiplier={multiplier} ? best_score so far: {best_score} -> new best')
                best_score = current_score
                best_move = possible_move

            else:
                logger.debug(
                    f'choice {moves_couter}: move: {possible_move}: current_score={current_score}, multiplier={multiplier} ? best_score so far: {best_score} -> ignore')

            # end recursion when alpha and beta say there is no better alternative to be found
            if maximizing_player:
                if best_score >= beta:
                    logger.debug(f"stop evaluating because best_score>= beta ( {best_score} >= {beta} )")
                    logger.pop_context()
                    return (best_score, best_move)
                alpha = max(alpha, best_score)
            else:
                if best_score <= alpha:
                    logger.debug(f"stop evaluating because best_score<= alpha ( {best_score} <= {alpha} )")
                    logger.pop_context()
                    return (best_score, best_move)
                beta = min(beta, best_score)

        logger.debug(f'END: best move {best_move}: score={best_score}')
        logger.pop_context()
        return (best_score, best_move)