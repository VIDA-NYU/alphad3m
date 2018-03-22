from __future__ import print_function
import sys
import os
from copy import deepcopy
sys.path.append('..')
from Game import Game
from .PipelineLogic import Board
import numpy as np
from random import uniform
import nn_evaluation
from d3m_metadata.problem import parse_problem_description
from d3m_ta2_nyu.metafeatures.dataset import compute_metafeatures

class PipelineGame(Game):
    def __init__(self, m, args=None):
        self.args = args
        self.evaluations = {}
        self.curr_evaluations = {}
        self.problem_features = parse_problem_description(self.args['problem_path']+'/problemDoc.json')
        self.problem = self.problem_features['problem']['task_type'].unparse().upper()
        self.dataset_metafeatures = list(compute_metafeatures(os.path.join(self.args['dataset_path'],'tables','learningData.csv')).values())
        #print(self.dataset_metafeatures)
        self.n = m + len(self.dataset_metafeatures)
        self.m = m
                
    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n, self.problem)
        count = 0
        for i in range(self.m, self.n):
            b.pieces[i][1] = self.dataset_metafeatures[count]
            count += 1
        return b.pieces+b.previous_moves 

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.m)

    def getActionSize(self):
        # return number of actions
        return len(Board.getPrimitives(self.problem).values())+1

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
        legalMoves =  b.get_legal_moves(self.problem)
        return np.array(legalMoves)

    def getEvaluation(self, board):
        primitive = board[0:self.m][self.m-1][1]
        if primitive == 0:
            self.curr_evaluations = {}
            return 0.0
        #print('EVALUATE PIPELINE', primitive)
        eval_val = self.evaluations.get(primitive)
        #eval_val = self.curr_evaluations.get(primitive)
        if eval_val is None:
            eval_val = self.evaluations.get(primitive)
            if eval_val is None:
                pipeline = ['dsbox.datapreprocessing.cleaner.KNNImputation','dsbox.datapreprocessing.cleaner.Encoder']
                for key, value in Board.getPrimitives(self.problem).items():
                    if primitive == value:
                        pipeline.append(key)
                #print('PIPELINE ', pipeline)
                eval_val = nn_evaluation.evaluate_pipeline_from_strings(pipeline, 'ALPHAZERO', self.args['dataset_path'], self.args['problem_path'])
                if eval_val is None:
                    eval_val = float('inf')
                self.evaluations[primitive] = eval_val
            #self.curr_evaluations[primitive] = eval_val
            
        return eval_val
    
    def getGameEnded(self, board, player, eval_val=None):
        # return 0 if not ended, 1 if x won, -1 if x lost
        # player = 1
        #print('\n\nEVALUATIONS ', self.evaluations)
        #if len(self.curr_evaluations) > 0:
        if len(self.evaluations) > 0:            
            #win_threshold = sorted(list(self.curr_evaluations.values()))[-1]
            win_threshold = sorted(list(self.evaluations.values()))[-1]
            b = Board(self.n, self.problem, win_threshold)
        else:
            b = Board(self.n, self.problem)
        b.pieces = board[0:self.n]
        b.previous_moves = board[self.n:]
        eval_val = self.getEvaluation(board)
        #print('\nEVAL', eval_val)
        #self.display(board)        
        if b.findWin(player, eval_val):
            #print('EVALUATIONS ', self.curr_evaluations)
            #print('findwin',player)
            return 1
        if b.findWin(-player, eval_val):
            #print('EVALUATIONS ', self.curr_evaluations)
            #print('findwin',-player)
            return -1
        if b.has_legal_moves():
            return 0

        return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return deepcopy(board)

    def stringRepresentation(self, board):
    	# 3x3 numpy array (canonical board)
    	return np.array(board[0:self.n]).tostring()+np.array(board[self.n:]).tostring()

    def getTrainExamples(self, board, pi):
        #print('LENGTH PI ', len(pi))
        assert(len(pi) == self.getActionSize())  # 1 for pass
        return [(board[0:self.n], pi)]
 
    def display(self, b):
        n = self.m
        board = b[0:n]
        for y in range(n):
            print (y,"|",end="")
        print("")
        print(" -----------------------")
        for y in range(n):
            print(y, "|",end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                if piece != 0: print(Board.getPrimitive(piece)+" ",end="")
                else:
                    if x==n:
                        print("-",end="")
                    else:
                        print("- ",end="")
            print("|")

        print("   -----------------------")
