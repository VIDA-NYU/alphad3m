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

class Board():
    PRIMITIVES_DECODE = ['None', 'E', 'I', 'S', 'K', 'D','NB', 'T', 'LO', 'LI', 'BR', 'L', 'R', 'LA']
    PRIMITIVES = {
        'PREPROCESSING':{
            'dsbox.datapreprocessing.cleaner.Encoder': 1,
            'dsbox.datapreprocessing.cleaner.KNNImputation': 2
        },
        'CLASSIFICATION': {
            'sklearn.svm.classes.LinearSVC': 3,
            'sklearn.neighbors.classification.KNeighborsClassifier': 4,
            'sklearn.tree.tree.DecisionTreeClassifier':5,
            'sklearn.naive_bayes.MultinomialNB':6,
            'sklearn.ensemble.forest.RandomForestClassifier':7,
            'sklearn.linear_model.logistic.LogisticRegression':8,
        },
        'REGRESSION': {
            'sklearn.linear_model.base.LinearRegression': 9,
            'sklearn.linear_model.bayes.BayesianRidge':10,
            'sklearn.linear_model.coordinate_descent.LassoCV':11,
            'sklearn.linear_model.ridge.Ridge': 12,
            'sklearn.linear_model.least_angle.Lars': 13,
        },
    }

    def __init__(self, m=30, problem='CLASSIFICATION', win_threshold=0.6):
        "Set up initial board configuration."
        
        self.m = m #Number of metafeatures
        self.p = len(self.PRIMITIVES['PREPROCESSING'].values()) + 1 #Length of pipeline
        
        # Create the empty board array.
        self.pieces_m = [0] * self.m
        self.pieces_p = [0] * self.p
        self.win_threshold = win_threshold
        self.problem = problem
        self.num_preprocessors = len(self.PRIMITIVES['PREPROCESSING'].values())
        self.num_estimators = len(self.PRIMITIVES[problem].values())
        self.valid_moves = self._compute_valid_moves(self.PRIMITIVES['PREPROCESSING'].values(), self.PRIMITIVES[problem].values())
        self.previous_moves = [0] * len(self.valid_moves)

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces_p[index]

    def findWin(self, player, eval_val=None):
        """Find win of the given color in row, column, or diagonal
        (1 for x, -1 for o)"""
        #print(self[0:])
        if not any(self[0:]):
            return False
        if eval_val == float('inf'):
            return False

        return eval_val >= self.win_threshold

    def get_legal_moves(self, action, problem='CLASSIFICATION'):
        """Returns all the legal moves.
        """

        valid_moves = [1]* (len(self.valid_moves))
        #print('VM ', valid_moves)
        #print('PM ', self.previous_moves)
        valid_moves = np.bitwise_xor(self.previous_moves, valid_moves)

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
        self.pieces_p = s
        self.previous_moves[action] = 1

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
