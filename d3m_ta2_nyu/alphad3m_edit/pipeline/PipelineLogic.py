'''
Board class.
Board data:
  1=x, -1=o, 0=empty
  first dim is column , 2nd is row:
     pieces[1][2] is the square in column 2, row 3.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''

import numpy as np
from itertools import combinations
from d3m_ta2_nyu.common import SCORES_FROM_SCHEMA, SCORES_RANKING_ORDER

class Board():
    PRIMITIVES_DECODE = ['None', 'E', 'I', 'S', 'K', 'D','NB', 'T', 'LO', 'LI', 'BR', 'L', 'R', 'LA']
    PRIMITIVES = {
        'PREPROCESSING':{
            'd3m.primitives.dsbox.Encoder': 1,
            'd3m.primitives.dsbox.MeanImputation': 2
        },
        'CLASSIFICATION': {
            'd3m.primitives.sklearn_wrap.SKRandomForestClassifier':3,
            'd3m.primitives.sklearn_wrap.SKDecisionTreeClassifier':4,
            'd3m.primitives.featuretools_ta1.SKRFERandomForestClassifier':5,
            'd3m.primitives.classifier.RandomForest':6,
            'd3m.primitives.sklearn_wrap.SKMultinomialNB':7,
            'd3m.primitives.common_primitives.BayesianLogisticRegression': 9,
            'd3m.primitives.dsbox.CorexSupervised': 11,
            'd3m.primitives.lupi_svm': 13,
            'd3m.primitives.realML.TensorMachinesBinaryClassification': 14,
            'd3m.primitives.sklearn_wrap.SKPassiveAggressiveClassifier': 15,
            'd3m.primitives.sklearn_wrap.SKQuadraticDiscriminantAnalysis': 16,
            'd3m.primitives.sklearn_wrap.SKSGDClassifier': 18,
            'd3m.primitives.sklearn_wrap.SKSVC': 19

    },
        'REGRESSION': {
            'd3m.primitives.cmu.autonlab.find_projections.SearchNumeric': 20,
            'd3m.primitives.cmu.autonlab.find_projections.SearchHybridNumeric': 21,
            'd3m.primitives.featuretools_ta1.SKRFERandomForestRegressor':22,
            'd3m.primitives.sklearn_wrap.SKARDRegression': 25,
            'd3m.primitives.sklearn_wrap.SKDecisionTreeRegressor': 26,
            'd3m.primitives.sklearn_wrap.SKExtraTreesRegressor': 27,
            'd3m.primitives.sklearn_wrap.SKGaussianProcessRegressor': 28,
            'd3m.primitives.sklearn_wrap.SKLars': 31,
            'd3m.primitives.sklearn_wrap.SKLasso': 32,
            'd3m.primitives.sklearn_wrap.SKLassoCV': 33,
            'd3m.primitives.sklearn_wrap.SKLinearSVR': 34,
            'd3m.primitives.sklearn_wrap.SKPassiveAggressiveRegressor': 35,
            'd3m.primitives.sklearn_wrap.SKSGDRegressor': 36,
            'd3m.primitives.sklearn_wrap.SKRidge':39
        },
    }
    OPERATIONS = {0:'insert',
                  1:'delete',
                  2:'substitute'}

    def __init__(self, m=30, pipeline=None, problem='CLASSIFICATION', metric='f1macro', win_threshold=0.6):
        "Set up initial board configuration."
       
        self.m = m #Number of metafeatures
        self.p = len(self.PRIMITIVES['PREPROCESSING'].values()) + 1 #Length of pipeline
        self.o = len(self.OPERATIONS)
        # Create the empty board array.
        self.pieces_m = [0] * self.m
        self.pieces_p = [0] * self.p
        if not pipeline is None:
       	    if len(pipeline) == 1:
               pipeline = [0, 0] + pipeline
            elif len(pipeline) == 2:
               pipeline = [0] + pipeline
        else:
            pipeline = [0] * self.p
        self.pieces_p = pipeline
        self.pieces_o = [0] * self.o
        if 'error' in metric.lower():
           win_threshold = -1 * win_threshold
        self.win_threshold = win_threshold
        self.problem = problem
        self.num_preprocessors = len(self.PRIMITIVES['PREPROCESSING'].values())
        self.num_estimators = len(self.PRIMITIVES[problem].values())
        self.valid_moves = self._compute_valid_moves(self.PRIMITIVES['PREPROCESSING'].values(), self.PRIMITIVES[problem].values())
        self.previous_moves = [0] * len(self.valid_moves)

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces_p[index]

    @classmethod
    def get_pipeline_size(cls):
        return len(cls.PRIMITIVES['PREPROCESSING'].values()) + 1

    @classmethod
    def get_edit_operations_size(cls):
        return len(cls.OPERATIONS)

    def get_pipeline(self, board):
        return board[self.m:self.m+self.p]

    def get_previous_moves(self, board):
        return board[self.m+self.p+self.o:]

    def get_metafeatures(self, board):
        return board[0:self.m]

    def get_operation(self, board):
        return board[self.m+self.p:self.m+self.p+self.o]

    def findWin(self, player, metric, eval_val=None):
        """Find win of the given color in row, column, or diagonal
        (1 for x, -1 for o)"""
        #print(self[0:])
        if not any(self[0:]):
            return False
        if eval_val == float('inf'):
            return False

        if SCORES_RANKING_ORDER[SCORES_FROM_SCHEMA[metric]] < 0:
            return eval_val >= self.win_threshold
        else:
            return eval_val <= self.win_threshold

    def get_legal_moves(self, problem='CLASSIFICATION'):
        """Returns all the legal moves.
        """

        valid_moves = [0]*len(self.valid_moves)
        valid_moves = self._get_insert_moves(valid_moves)
        if any(self.pieces_p):
            valid_moves = self._get_substitue_moves(valid_moves)
        if np.where(np.asarray(self.pieces_p)>0)[0].shape[0] > 1:
            valid_moves = self._get_delete_moves(valid_moves)

        #print('VM ', valid_moves)
        #print('PM ', self.previous_moves)

        for i in range(0, len(self.previous_moves)):
            if self.previous_moves[i] == 1:
                valid_moves[i] = 0
        #valid_moves = self.previous_moves, valid_moves

        #if np.sum(valid_moves) == 0:
        #    valid_moves = []
        return valid_moves
        
    def has_legal_moves(self, problem='CLASSIFICATION'):
        return not all(self.previous_moves)

    #def _getMove(self, action):
    #   return [val for val in Board.PRIMITIVES[self.problem].values()][action]

    @classmethod
    def getPrimitives(cls, problem='CLASSIFICATION'):
        return cls.PRIMITIVES[problem]

    @classmethod
    def getPrimitive(cls, piece):
        return cls.PRIMITIVES_DECODE[piece]

    def execute_move(self, action, player):
        """Perform the given move on the board;
        color gives the color of the piece to play (1=x,-1=o)
        """
        s = self.valid_moves[action]
       
        if len(s) == 1:
            s = [0, 0] + s
        elif len(s) == 2:
            s = [0] + s
        pipeline = np.asarray((self.pieces_p))[np.where(np.asarray(self.pieces_p) > 0)[0]].tolist()
        if len(s) > len(pipeline):
            self.pieces_o[0] = 1
        elif len(s) < len(pipeline):
                self.pieces_o[1] = 1
        if len(s) == len(pipeline):
            self.pieces_o[2] = 1

        self.pieces_p = s
        self.previous_moves[action] = 1

    def _get_insert_moves(self, valid_moves):
        if any(self.pieces_p):
            pipeline_array = np.asarray((self.pieces_p))[np.where(np.asarray(self.pieces_p) > 0)[0]]
            N = pipeline_array.shape[0]
            pipeline_set = set(pipeline_array.tolist())
        else:
            pipeline_set = set()
            N = 0

        index = 0
        for move in self.valid_moves:
            if len(move) == (N+1):
                if len(set(move).difference(pipeline_set)) == 1:
                    valid_moves[index] = 1
            index = index + 1

        return valid_moves

    def _get_delete_moves(self, valid_moves):
        if any(self.pieces_p):
            pipeline_array = np.asarray((self.pieces_p))[np.where(np.asarray(self.pieces_p) > 0)[0]]
            N = pipeline_array.shape[0]
            pipeline_set = set(pipeline_array.tolist())
        else:
            pipeline_set = set()
            N = 0

        index = 0
        for move in self.valid_moves:
            if len(move) == (N - 1):
                if len(set(move).intersection(pipeline_set)) == N-1:
                    valid_moves[index] = 1
            index = index + 1

        return valid_moves

    def _get_substitue_moves(self, valid_moves):

        if any(self.pieces_p):
            pipeline_array = np.asarray((self.pieces_p))[np.where(np.asarray(self.pieces_p) > 0)[0]]
            N = pipeline_array.shape[0]
            pipeline_set = set(pipeline_array.tolist())
        else:
            pipeline_set = set()
            N = 0

        index = 0
        for move in self.valid_moves:
            if len(move) == N:
                if len(set(move).intersection(pipeline_set)) == (N-1):
                    valid_moves[index] = 1
            index = index + 1

        return valid_moves

    def _compute_valid_moves(self, pp, est):
        valid_moves = []
        for l in range(1, len(pp)+1):
            valid_moves = valid_moves + [list(c) for c in combinations(pp, l)]

        tmp = valid_moves
        valid_moves = []
        for i in range(0, len(tmp)):
            for e in est:
                valid_moves.append(tmp[i]+[e])
        return [[e] for e in est] + valid_moves
