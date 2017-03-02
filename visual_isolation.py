import sys, os, random, pygame
sys.path.append(os.path.join("objects"))
import IsolationSquare
from GameResources import *

import queue


from isolation import Board

class VisualBoard(object):

    theSquares = {}
    size_x = 100
    size_y = 100
    screen = None
    background_image = None
    clock = None

    def __init__(self, rows, columns, size_x=100, size_y=100):
        self.size_x = size_x
        self.size_y = size_y
        self.rows = rows
        self.columns = columns
        self.mode = (self.size_x * self.columns, self.size_y * self.rows)

        pygame.init()
        self.screen = pygame.display.set_mode(self.mode)
        self.background_image = pygame.image.load("./images/sudoku-board-bare.jpg").convert()
        self.clock = pygame.time.Clock()

        for x in range(self.columns):
            for y in range(self.rows):
                self.theSquares[(x,y)] = (IsolationSquare.IsolationSquare( "",self.size_x*x, self.size_y*y, x,y))


    def draw_board(self, board):

        pygame.event.pump()

        pygame.display.flip()
        pygame.display.update()
        self.clock.tick(5)

        # leave game showing until closed by user
        # while True:
        #     for event in pygame.event.get():
        #         print(event)
        #         if event.type == pygame.QUIT:
        #             pygame.quit()
        #             quit()


    def update_board(self, board):

        pygame.event.pump()
        self.screen.blit(self.background_image, (0, 0))

        for (x,y) in self.theSquares:
            self.theSquares[(x,y)].draw()
        #        theSquares.append(IsolationSquare.IsolationSquare( "",size_x*x, size_y*y, False, x,y))

        if board == None:
            pass
        else:
            for(x,y) in board.get_blank_spaces():
                self.theSquares[(x,y)].make_clear()

            self.screen.blit(self.background_image, (0, 0))
            for (x,y) in self.theSquares:
                self.theSquares[(x,y)].draw()

            p1 = board.get_player_location(board.__player_1__)
            # print("position: " + p1)
            if p1 is not None:
                self.theSquares[p1].make_A()
            p2 = board.get_player_location(board.__player_2__)
            if p2:
                self.theSquares[p2].make_B()

        pygame.display.flip()
        pygame.display.update()
        self.clock.tick(5)

    def main_loop(self, q_in): #, q_out):

        EV_USER_UPDATE = pygame.USEREVENT + 1
        pygame.time.set_timer(EV_USER_UPDATE,100)

        done = False
        loop_counter = 0
        board_state = None
        while not done:

            print("visual_thread: main loop counter %d" % loop_counter)
            loop_counter = loop_counter + 1
            event_counter=0
            for event in pygame.event.get():
                event_counter = event_counter + 1
                print("visual_thread: event %d : %s" % (event_counter,event))
                # print(event)
                if event.type == pygame.QUIT:
                    done = True
                    break
                elif event.type == pygame.MOUSEBUTTONUP:
                    print("visual_thread: user clik ad %d %d" % event.pos)

                elif event.type == EV_USER_UPDATE:
                    print("visual_thread: time to update the board")

                    new_board_state = None
                    try:
                        new_board_state = q_in.get(False)
                        self.update_board(board_state)
                        board_state = new_board_state
                        print("visual_thread: updated visuals")
                    except queue.Empty:
                        print("visual_thread: no board update to read from queue")
                    # pygame.time.Clock.tick(40)
            self.clock.tick(5)
            pygame.event.pump()

        pygame.quit()
        return