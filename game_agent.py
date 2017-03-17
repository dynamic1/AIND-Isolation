"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import math
import sys




directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
              (1, -2), (1, 2), (2, -1), (2, 1)]

directions_2 = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
              (1, -2), (1, 2), (2, -1), (2, 1)]

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
    # utility = game.utility(player)
    # if utility:
    #     return utility

    # x, y = game.get_player_location(player)
    # delta = float((abs(3 - x) + abs(3 - y)) / 2)
    return float( len(game.get_legal_moves(player)) - 0.5 * len(game.get_legal_moves(game.get_opponent(player))))
    return float( len(game.get_legal_moves(player)) - 0.5 * len(game.get_legal_moves(game.get_opponent(player)))) - delta


def custom_score_adv(game, player):
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

    x, y = game.get_player_location(player)
    # delta = float((abs(3 - x) + abs(3 - y)) / 2)
    return float(len(game.get_legal_moves(player)) - 0.7 * len(game.get_legal_moves(game.get_opponent(player))))

def custom_score_adv_opt(game, player):
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
    rp, cp = game.__last_player_move__[player]
    ro, co = game.__last_player_move__[game.get_opponent(player)]

    score = 0
    for dr, dc in directions:
        if 0 <= rp+dr < game.height and 0 <= cp+dc < game.width and \
           game.__board_state__[rp+dr][cp+dc] == game.BLANK:
            score += 1
        if 0 <= ro+dr < game.height and 0 <= co+dc < game.width and \
           game.__board_state__[ro+dr][co+dc] == game.BLANK:
            score += (-.7)

    return float(score)


def custom_score_center(game, player):
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

    x, y = game.get_player_location(player)
    delta = float(abs(3 - x) + abs(3 - y))
    return float(len(game.get_legal_moves(player)) - delta /3)

def custom_score_margins(game, player):
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

    x, y = game.get_player_location(player)
    delta = float(abs(3 - x) + abs(3 - y))
    return float(len(game.get_legal_moves(player)) + delta /3)



def custom_score_coop(game, player):
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

    x, y = game.get_player_location(player)
    # delta = float((abs(3 - x) + abs(3 - y)) / 2)
    return float(len(game.get_legal_moves(player)) + 0.5 * len(game.get_legal_moves(game.get_opponent(player))))

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
                 iterative=True, method='minimax', timeout=20.):
        self.max_depth_reached = 0
        self.stats_by_depth = {}
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.current_depth_nodes = 0
        #xalex
        self.moves_by_depth = []

    def my_function_name(self):
        """
        :return: name of caller
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

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        # best_score = -math.inf
        if len(legal_moves) < 1:
            return (-1,-1)

        best_move = legal_moves[0]
        current_depth = 0 if self.iterative else self.search_depth
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            forecast_couter = 0

            local_best_move = (-1,-1)
            local_best_score = -math.inf

            reiterate = True
            self.current_depth_nodes = 0

            while reiterate:
                self.current_depth_nodes = 0
                if len(legal_moves) <1:
                    return best_move

                for move_idx, possible_move in enumerate(legal_moves):
                    time_left_at_start = self.time_left()
                    possible_game = game.forecast_move(possible_move)
                    forecast_couter += 1
                    # possible_game = game.forecast_move(possible_move)
                    # possible_score, recommended_move = self.minimax(possible_game,1,True)
                    if self.method == 'minimax':
                        possible_score, recommended_move = self.minimax(possible_game,current_depth, False )
                    else:
                        # possible_score, recommended_move = self.alphabeta(possible_game, current_depth, -math.inf, math.inf, False)
                        possible_score, recommended_move = self.alphabeta(possible_game, current_depth, -math.inf, math.inf, False)
                    if possible_score > local_best_score:
                        local_best_move = possible_move
                        local_best_score = possible_score

                    time_left_after = self.time_left()
                    self.stats_by_depth[current_depth] = (best_move, local_best_score, round(time_left_after), round(time_left_at_start- time_left_after), self.current_depth_nodes)

                if not self.iterative:
                    return local_best_move

                best_move = local_best_move
                current_depth += 1

        except Timeout:
            # Handle any actions required at timeout, if necessary
            for i_d,(i_move, i_score, i_time_left, i_duration, i_nodes) in self.stats_by_depth.items():
                pass
            return best_move

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

        best_move=(-1,-1)
        best_score = (- math.inf) if maximizing_player else math.inf

        scoring_player = game.active_player if maximizing_player else game.inactive_player
        """
        if maximizing_player:
            my_player = game.active_player
            other_player = game.inactive_player
        else:
            my_player = game.inactive_player
            other_player = game.active_player
        """
        current_score = self.score(game, scoring_player)
        current_move = game.get_player_location(game.inactive_player)

        # if depth is 0, will evaluate current branch using just the self.score function, no further recursion
        if depth ==0:
            return (current_score, current_move)

        # will check for legal moves and recurse to evaluate each legal move
        legal_moves = game.get_legal_moves(game.active_player)

        # if there are no available moves
        if len(legal_moves) == 0:
            if (game.is_loser(scoring_player)):
                return (-math.inf, current_move)

            return (math.inf, current_move)


        moves_couter = 0
        forecast_couter = 0
        for possible_move in legal_moves:

            forecast_couter += 1
            possible_game = game.forecast_move(possible_move)
            moves_couter += 1

            current_score, optimal_move = self.minimax(possible_game, depth - 1, not maximizing_player)

            #  record new best move and score
            if ( maximizing_player and (current_score > best_score)) or ( ( not maximizing_player ) and (current_score < best_score)):
                best_score = current_score
                best_move = possible_move

            else:
                pass


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

        best_move=(-1,-1)
        # multiplier = 1 if maximizing_player else -1
        best_score = (- math.inf) if maximizing_player else math.inf

        scoring_player = game.active_player if maximizing_player else game.inactive_player

        """
        if depth is 0, will evaluate current branch using just the self.score function, no further recursion
        """
        current_move = game.get_player_location(game.inactive_player)

        if depth ==0:
            current_score = self.score(game, scoring_player)

            return (current_score, current_move)
            # return (-100, (-1,-1))

        """
        will check for legal moves and recurse to evaluate each legal move
        """
        legal_moves = game.get_legal_moves(game.active_player)

        # if there are no available moves
        if len(legal_moves)==0:
            if(game.is_loser(scoring_player)):
                return (-math.inf, (-1,-1))
                # return (-math.inf ,(-1,-1))

            return (math.inf , current_move)

        moves_counter = 0
        forecast_couter = 0

        """
        try to sort possible moves so that I would investigate best first
        """
        """
        l_moves = []
        for possible_move in legal_moves:
            forecast_couter = forecast_couter + 1
            possible_game = game.forecast_move(possible_move)
            possible_score = self.score(possible_game,scoring_player)
            l_moves.append({'move':possible_move, 'score':possible_score, 'game':possible_game})

        sort_before_alphabeta = True
        sort_before_alphabeta = False
        if sort_before_alphabeta:
            l_sorted_moves = sorted(l_moves, key=lambda x: x['score'], reverse=True)
        else:
            l_sorted_moves = l_moves

        for branch in l_sorted_moves:
            pass

        # for possible_move in legal_moves:
        for branch in l_sorted_moves:

            self.current_depth_nodes += 1
            possible_game = branch['game']
            possible_move = branch['move']
            moves_counter = moves_counter+1
            """

        for possible_move in legal_moves:
            forecast_couter += 1
            possible_game = game.forecast_move(possible_move)
            # possible_score = self.score(possible_game, scoring_player)
            # l_moves.append({'move': possible_move, 'score': possible_score, 'game': possible_game})
            moves_counter += 1


            current_score, optimal_move = self.alphabeta(possible_game, depth - 1, alpha, beta, not maximizing_player)

            if ( maximizing_player and (current_score > best_score) ) or ( (not maximizing_player) and (current_score<best_score) ):
                best_score = current_score
                best_move = possible_move

            else:
                pass


            """
            end recursion when alpha and beta say there is no better alternative to be found
            """

            if maximizing_player:
                alpha = max(alpha, best_score)

            else:
                beta = min(beta, best_score)

            if alpha >= beta:
                return best_score, best_move

        return best_score, best_move
