from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .PipelineLogic import Board
import numpy as np
from random import uniform
import nn_evaluation
from d3m_metadata.problem import parse_problem_description

class PipelineGame(Game):
    def __init__(self, n, args=None):
        self.n = n
        self.args = args
        self.evaluations = {}
        problem_features = parse_problem_description(self.args['problem_path']+'/problemDoc.json')
        self.problem = problem_features['problem']['task_type'].unparse().upper()
        
    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n, self.problem)
        return b.pieces+b.previous_moves

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return len(Board.PRIMITIVES[self.problem].values())+1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = Board(self.n, self.problem)
        b.pieces = board[0:self.n]
        b.previous_moves = board[self.n:]
        b.execute_move(action, player)
        return (b.pieces+b.previous_moves, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        b = Board(self.n,self.problem)
        b.pieces = board[0:self.n]
        b.previous_moves = board[self.n:]
        legalMoves =  b.get_legal_moves()
        return np.array(legalMoves)

    def getEvaluation(self, board):
        # primitive = np.array(board[0:self.n]).tostring()
        # eval_val = self.evaluations.get(primitive)
        # if eval_val is None:
        #     eval_val = uniform(0.7, 0.9)
        #     self.evaluations[primitive] = eval_val
        primitive = board[0:self.n][self.n-1][1]
        if primitive == 0:
            return 0.0
        print('EVALUATE PIPELINE', primitive)

        eval_val = self.evaluations.get(primitive)
        if eval_val is None:
            pipeline = ['dsbox.datapreprocessing.cleaner.KNNImputation','dsbox.datapreprocessing.cleaner.Encoder']
            for key, value in Board.PRIMITIVES[self.problem].items():
                if primitive == value:
                    pipeline.append(key)
            result = nn_evaluation.evaluate_pipeline_from_strings(pipeline, 'ALPHAZERO', self.args['dataset_path'], self.args['problem_path'])
            eval_val = result['F1_MACRO']
            self.evaluations[primitive] = eval_val
        return eval_val
    
    def getGameEnded(self, board, player, eval_val=None):
        # return 0 if not ended, 1 if x won, -1 if x lost
        # player = 1
        b = Board(self.n, self.problem)
        b.pieces = board[0:self.n]
        b.previous_moves = board[self.n:]
        eval_val = self.getEvaluation(board)
        print('\nEVAL', eval_val)
        self.display(board)        
        if b.findWin(player, eval_val):
            print('findwin',player)
            return 1
        if b.findWin(-player, eval_val):
            print('findwin',-player)
            return -1
        if b.has_legal_moves():
            return 0
        
        return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        #print(board)
        return board

    def stringRepresentation(self, board):
    	# 3x3 numpy array (canonical board)
    	return np.array(board[0:self.n]).tostring()+np.array(board[self.n:]).tostring()

    def display(self, b):
        n = self.n
        board = b[0:n]
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
