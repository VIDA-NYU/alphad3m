#from __future__ import print_function

import logging
import sys
import os
import pickle
import math
import numpy as np
from scipy.special import comb
from random import uniform
from copy import deepcopy
import traceback
sys.path.append('..')

from ..Game import Game
from .PipelineLogic import Board

from d3m.metadata.problem import parse_problem_description

PROBLEM_TYPES = {'CLASSIFICATION': 1,
                 'REGRESSION': 2}
DATA_TYPES = {'TABULAR': 1}

logger = logging.getLogger(__name__)

class PipelineGame(Game):
    def __init__(self, args, pipeline, eval_pipeline, compute_metafeatures):
        self.steps = 0
        self.eval_pipeline = eval_pipeline
        self.pipeline = pipeline
        self.args = args
        self.evaluations = {}
        self.curr_evaluations = {}
        self.problem = self.args['problem']['about']['taskType'].upper()
        self.metric = self.args['problem']['inputs']['performanceMetrics'][0]['metric']
        self.dataset_metafeatures = None
        metafeatures_path = args.get('metafeatures_path')
        if not metafeatures_path is None:
            metafeatures_file = os.path.join(metafeatures_path, args['dataset'] + '_metafeatures.pkl')
            if os.path.isfile(metafeatures_file):
                m_f = open(metafeatures_file, 'rb')
                self.dataset_metafeatures = pickle.load(m_f)[args['dataset']]

        if self.dataset_metafeatures is None:
            self.dataset_metafeatures = compute_metafeatures.compute_metafeatures('AlphaD3M_compute_metafeatures')
        self.dataset_metafeatures = list(np.nan_to_num(np.asarray(self.dataset_metafeatures)))
        self.p = Board.get_pipeline_size()
        self.m = len(self.dataset_metafeatures)+2
        self.o = Board.get_edit_operations_size()
        self.action_size = 0

                
    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.m, self.pipeline, self.problem, self.metric)
        self.p = b.p
        self.o = b.o
        for i in range(0, len(self.dataset_metafeatures)):
            b.pieces_m[i] = self.dataset_metafeatures[i]
        b.pieces_m[i] = DATA_TYPES['TABULAR']
        i = i+1
        b.pieces_m[i] = PROBLEM_TYPES[self.problem]
        self.action_size = len(b.valid_moves)
        return b.pieces_m + b.pieces_p + b.pieces_o + b.previous_moves

    def getBoardSize(self):
        # (a,b) tuple
        return self.m + self.p + self.o

    def getActionSize(self):
        # return number of actions
        pp = len(Board.PRIMITIVES['PREPROCESSING'])
        est = len(Board.PRIMITIVES[self.problem])
        action_size = 0
        for i in range(1, pp + 1):
            action_size = action_size + comb(pp, i)
        action_size = action_size * est  # Account for the estimators
        action_size = action_size + est  # When no primitives are used

        return int(action_size)

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = Board(self.m, self.pipeline, self.problem, self.metric)
        b.pieces_m = b.get_metafeatures(board)
        b.previous_moves = b.get_previous_moves(board)
        b.execute_move(action, player)
        return (b.pieces_m+b.pieces_p+b.pieces_o+b.previous_moves, -player)

    def getValidMoves(self, board, player, ignore_prev_moves=False):
        # return a fixed size binary vector
        b = Board(self.m, self.pipeline, self.problem, self.metric)
        b.pieces_m = b.get_metafeatures(board)
        b.pieces_p = b.get_pipeline(board)
        if not ignore_prev_moves:
            b.previous_moves = b.get_previous_moves(board)
        legalMoves =  b.get_legal_moves(self.problem)
        logger.info("%s", b.pieces_p)
        logger.info("%s", [b.valid_moves[i] for i in range(0, len(legalMoves)) if legalMoves[i] == 1])
        return np.array(legalMoves)

    def getEvaluation(self, board):
        valid_p_names = list(Board.getPrimitives('PREPROCESSING').keys())+list(Board.getPrimitives(self.problem).keys())
        valid_p_enum = list(Board.getPrimitives('PREPROCESSING').values())+list(Board.getPrimitives(self.problem).values())
        b = Board(self.m, self.pipeline, self.problem, self.metric)
        pipeline_enums = b.get_pipeline(board)
        pipeline = [valid_p_names[valid_p_enum.index(board[i])] for i in range(self.m, self.m+self.p) if not board[i] == 0]
        if not any(pipeline):
            self.curr_evaluations = {}
            return 0.0
        eval_val = self.evaluations.get(",".join(pipeline))
        if eval_val is None:
            self.steps = self.steps + 1
            logger.info("%s", pipeline)
            eval_val = self.eval_pipeline(pipeline, 'AlphaD3M_Edit_eval')
            logger.info("Evaluation = %s", eval_val)
            if eval_val is None:
                eval_val = float('inf')
            self.evaluations[",".join(pipeline)] = eval_val
        return eval_val
    
    def getGameEnded(self, board, player, eval_val=None):
        # return 0 if not ended, 1 if x won, -1 if x lost
        # player = 1

        if len(self.evaluations) > 0:
            sorted_evals = sorted([eval for eval in list(self.evaluations.values()) if eval != float('inf')])
            if len(sorted_evals) > 0:
                if 'error' in self.metric.lower():
                   win_threshold = sorted_evals[0]
                else:
                   win_threshold = sorted_evals[-1]
                b = Board(self.m, self.pipeline, self.problem, self.metric, win_threshold)
            else:
                b = Board(self.m, self.pipeline, self.problem, self.metric)
        else:
            b = Board(self.m, self.pipeline, self.problem, self.metric)
        b.pieces_m = b.get_metafeatures(board)
        b.pieces_p = b.get_pipeline(board)
        b.previous_moves = b.get_previous_moves(board)
        eval_val = self.getEvaluation(board)
        if b.findWin(player, self.metric, eval_val):
            logger.info("EVALUATIONS %s", self.evaluations)
            logger.info('findwin %s',player)
            return 1
        if b.findWin(-player, self.metric, eval_val):
            logger.info("EVALUATIONS %s", self.evaluations)
            logger.info('findwin %s', player)
            return -1
        if b.has_legal_moves():
            return 0

        return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return deepcopy(board)

    def stringRepresentation(self, board):
        # 3x3 numpy array (canonical board)
        return np.asarray(board[:self.m+self.p+self.o]).tostring()

    def getTrainExamples(self, board, pi):
        assert(len(pi) == self.getActionSize())  # 1 for pass
        eval_val = self.getEvaluation(board)
        if 'error' in self.metric.lower():
            return (board[:self.m+self.p+self.o], pi, eval_val if eval_val!= float('inf') and eval_val <= math.pow(10,15) else math.pow(10,15))
        else:
            return (board[:self.m+self.p+self.o], pi, eval_val if eval_val != float('inf') else 0)

    def display(self, b):
        n = self.p
        #print(b)
        board = b[self.m:self.m+self.p]
        logger.info(" -----------------------")
        logger.info("PIPELINE %s", '|'.join(str(e) for e in board))
        logger.info(" -----------------------")
