"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math


"""
xalex
use logging for debug
"""
import logging
import logging.config

logging.config.fileConfig('logging.conf')

# create logger
logger = logging.getLogger('simpleExample')

## expale useage of logger
# logger.debug('debug message')
# logger.info('info message')
# logger.warning('warn message')
# logger.error('error message')
# logger.critical('critical message')



class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


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

    # TODO: finish this function!
    raise NotImplementedError


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

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        best_score = -math.inf
        best_move = (-1,-1)
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            forecast_couter = 0
            for possible_move in legal_moves:
                possible_game = game.forecast_move(possible_move)
                forecast_couter = forecast_couter + 1
                possible_game = game.forecast_move(possible_move)
                logger.debug(f'DEPTH={depth} GET_MOVE doing forecast counter={forecast_couter}, move={possible_move}')
                possible_score, recommended_move = self.minimax(possible_game,1,True)
                if possible_score > best_score:
                    best_move = possible_move
                    best_score = possible_score
            return best_move

        except Timeout:
            # Handle any actions required at timeout, if necessary
            logger.warning("Timeout occured in get_move()")
            pass

        # Return the best move from the last completed search iteration
        logger.warning("Not implemented in get_move()")
        raise NotImplementedError

    def minimax(self, game, depth, maximizing_player=True):
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

        # TODO: finish this function!
        # raise NotImplementedError

        my_type = "MAX" if maximizing_player else "MIN"
        logger.debug(f'DEPTH={depth} MINIMAX start {my_type} depth={depth}' )

        print(game.to_string())

        best_move=(-1,-1)
        multiplier = 1 if maximizing_player else -1
        best_score = (- math.inf) #if maximizing_player else math.inf


        my_player = game.active_player if maximizing_player else game.inactive_player
        # for possible_move in game.get_legal_moves(my_player):
        #     logger.debug(f'possible move: {possible_move}')


        moves_couter = 0
        forecast_couter = 0;
        for possible_move in game.get_legal_moves(my_player):
            forecast_couter = forecast_couter + 1
            possible_game = game.forecast_move(possible_move)
            logger.debug(f'DEPTH={depth} MIMIMAX doing forecast counter={forecast_couter}, move={possible_move}, new_counter={possible_game.counts[1]}')
            moves_couter = moves_couter+1
            if depth == 1:
                # current_score = self.score(possible_game, possible_game.active_player if maximizing_player else possible_game.inactive_player)
                current_score = self.score(possible_game, possible_game.inactive_player if maximizing_player else possible_game.active_player)
                #logger.debug(f'move: {possible_move}: current_score={current_score}')

            else:
                logger.debug(f'DEPTH={depth} MINIMAX calling recursive, depth = {depth-1}, maximizing={not maximizing_player} to evaluate possible_move {possible_move}')
                current_score, optimal_move = self.minimax(possible_game, depth - 1, not maximizing_player)
                logger.debug(f'DEPTH={depth} move: {possible_move}: current_score={current_score}')
                pass

            if current_score* multiplier > best_score:
                logger.debug(
                    f'DEPTH={depth} choise {moves_couter}: move: {possible_move}: current_score={current_score}, multiplier={multiplier} ? best_score so far: {best_score} -> new best')
                best_score = current_score
                best_move = possible_move

            else:
                logger.debug(
                    f'DEPTH={depth} choise {moves_couter}: move: {possible_move}: current_score={current_score}, multiplier={multiplier} ? best_score so far: {best_score} -> ignore')

        logger.debug(f'DEPTH={depth} minimax END: best move {best_move}: score={best_score}')
        return (best_score, best_move)
        # return (0,(1,5))



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

        # TODO: finish this function!
        raise NotImplementedError
