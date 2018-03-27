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


class Board():
    PRIMITIVES_DECODE = ['None', 'E', 'I', 'S', 'K', 'D','NB', 'T', 'LO', 'LI', 'BR', 'L', 'R', 'LA']
    PRIMITIVES = {
        'ENCODER':{
            'dsbox.datapreprocessing.cleaner.Encoder': 1
        },
        'IMPUTATION': {
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

    def __init__(self, n=3, problem='CLASSIFICATION', win_threshold=0.6):
        "Set up initial board configuration."

        self.n = n
        self.m = 3
        # Create the empty board array.
        self.pieces = [None] * self.n
        for i in range(self.n):
            self.pieces[i] = [0] * self.m
        self[0][1] = 1
        self[1][1] = 2
        self.previous_moves = [0]*len(self.PRIMITIVES[problem].values())
        self.win_threshold = win_threshold
        self.problem = problem
        
    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces[index]

    def findWin(self, player, eval_val=None):
        """Find win of the given color in row, column, or diagonal
        (1 for x, -1 for o)"""
        if self[self.m-1][1] == 0:
            return False
        if eval_val == float('inf'):
            return False

        #print('\n\nPLAYER ', player)
        #print('EVAL', eval_val)
        #print('WIN THRESHOLD ', self.win_threshold)
        #print('FIND WIN ', eval_val >= self.win_threshold)
        return eval_val >= self.win_threshold

    def get_legal_moves(self, problem='CLASSIFICATION'):
        """Returns all the legal moves.
        """
        valid_moves = [1]*(len(self.PRIMITIVES[problem].values()))+[0]
        #print('PREVIOUS MOVES ', self.previous_moves)
        valid_moves = np.bitwise_xor(self.previous_moves+[0], valid_moves)
        #print('VALID MOVES ', valid_moves)
        if np.sum(valid_moves) == 0:
            valid_moves[-1] = 1
        return valid_moves
        
    def has_legal_moves(self, problem='CLASSIFICATION'):
        return len(self.previous_moves) < len(self.PRIMITIVES[problem].values())

    def _getMove(self, action):
        return [val for val in Board.PRIMITIVES[self.problem].values()][action]

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

        # Add the piece to the empty square.
        # print(move)
        # print(self[x][y],color)
        x = 2
        y = 1
        self[x][y] = self._getMove(action)
        self.previous_moves[action] = 1

