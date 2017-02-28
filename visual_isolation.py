import sys, os, random, pygame
sys.path.append(os.path.join("objects"))
import IsolationSquare
from GameResources import *

from isolation import Board

digits = '123456789'
rows = 'ABCDEFGHI'

class VisualBoard(object):

    theSquares = {}
    size_x = 100
    size_y = 100
    screen = None
    background_image = None
    clock = None

    def __init__(self, board, size_x=100, size_y=100):
        self.size_x = size_x
        self.size_y = size_y

        pygame.init()
        self.screen = pygame.display.set_mode((self.size_x * board.width, self.size_y * board.height))
        self.background_image = pygame.image.load("./images/sudoku-board-bare.jpg").convert()
        self.clock = pygame.time.Clock()

        for x in range(board.width):
            for y in range(board.height):
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

        for(x,y) in board.get_blank_spaces():
            self.theSquares[(x,y)].make_clear()

        self.screen.blit(self.background_image, (0, 0))
        for (x,y) in self.theSquares:
            self.theSquares[(x,y)].draw()

        p1 = board.get_player_location(board.active_player)
        # print("position: " + p1)
        if p1 is not None:
            self.theSquares[p1].make_A()
        p2 = board.get_player_location(board.inactive_player)
        if p2:
            self.theSquares[p2].make_B()

        pygame.display.flip()
        pygame.display.update()
        self.clock.tick(5)

