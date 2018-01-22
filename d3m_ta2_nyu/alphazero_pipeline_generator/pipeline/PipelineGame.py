from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .PipelineLogic import Board
import numpy as np
from random import uniform

class PipelineGame(Game):
    def __init__(self, n):
        self.n = n
        self.previous_moves = []
        self.evaluations = {}
        
    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self, problem='CLASSIFICATION'):
        # return number of actions
        return len(Board.PRIMITIVES[problem].values())+1

    def clearPrevMoves(self):
        self.previous_moves = []

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        #if action == self.getActionSize():
        #    return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        b.previous_moves = self.previous_moves
        move = b.getMove(action)
        b.execute_move(move, player)
        self.previous_moves = b.previous_moves
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        b = Board(self.n)
        b.pieces = np.copy(board)
        b.previous_moves = self.previous_moves
        legalMoves =  b.get_legal_moves()
        return np.array(legalMoves)

    def getEvaluation(self, board):
        primitive = board[2][1]
        eval_val = self.evaluations.get(primitive)
        if eval_val is None:
            eval_val = uniform(0.7, 0.9)
            self.evaluations[primitive] = eval_val
        return eval_val
    
    def getGameEnded(self, board, player, eval_val=None):
        # return 0 if not ended, 1 if x won, -1 if x lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)
        b.previous_moves = self.previous_moves
        #print('CHECK GAME ENDED')
        eval_val = self.getEvaluation(board)
        print('\nEVAL', eval_val)
        self.display(board)        
        if b.findWin(player, eval_val):
            print('findwin',player)
            self.clearPrevMoves()
            return 1
        if b.findWin(-player, eval_val):
            print('findwin',-player)
            self.clearPrevMoves()            
            return -1
        if b.has_legal_moves():
            return 0
        
        return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        #print(board)
        return board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        #print('\n\n\nASSERT ', len(pi), ' ', self.getActionSize())
        assert(len(pi) == self.getActionSize())  # 1 for pass
        l = [(board, pi)]
        return l

    def stringRepresentation(self, board):
    	# 3x3 numpy array (canonical board)
    	return board.tostring()

    def display(self, board):
        n = board.shape[0]

        for y in range(n):
            print (y,"|",end="")
        print("")
        print(" -----------------------")
        for y in range(n):
            print(y, "|",end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                if piece != 0: print(Board.PRIMITIVES_DECODE[piece]+" ",end="")
                else:
                    if x==n:
                        print("-",end="")
                    else:
                        print("- ",end="")
            print("|")

        print("   -----------------------")
