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
            'sklearn.neural_network.MLPClassifier':9,
            'sklearn.svm.SVC':10,
            'sklearn.gaussian_process.GaussianProcessClassifier':11,
            'sklearn.gaussian_process.kernels.RBF':12,
            'sklearn.tree.DecisionTreeClassifier':13,
            'sklearn.ensemble.AdaBoostClassifier':14,
            'sklearn.naive_bayes.GaussianNB':15,
            'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis':16,
            'sklearn.linear_model.SGDClassifier':17

    },
        'REGRESSION': {
            'sklearn.linear_model.ARDRegression': 18,
            'sklearn.linear_model.BayesianRidge': 19,
            'sklearn.linear_model.ElasticNet': 20,
            #'sklearn.linear_model.ElasticNetCV': 21,
            'sklearn.linear_model.HuberRegressor': 22,
            'sklearn.linear_model.Lars': 23,
            #'sklearn.linear_model.LarsCV': 24,
            'sklearn.linear_model.Lasso': 25,
            #'sklearn.linear_model.LassoCV': 26,
            'sklearn.linear_model.LassoLars': 27,
            #'sklearn.linear_model.LassoLarsCV': 28,
            'sklearn.linear_model.LassoLarsIC': 29,
            'sklearn.linear_model.LinearRegression': 30,
            'sklearn.linear_model.PassiveAggressiveRegressor': 31,
            'sklearn.linear_model.RANSACRegressor': 32,
            'sklearn.linear_model.Ridge': 33,
            'sklearn.linear_model.RidgeCV': 34,
            'sklearn.linear_model.SGDRegressor': 35,
            'sklearn.linear_model.TheilSenRegressor': 36
        },
    }
    OPERATIONS = {0:'insert',
                  1:'delete',
                  2:'substitute'}

    def __init__(self, m=30, pipeline=[1,2,17], problem='CLASSIFICATION', win_threshold=0.6):
        "Set up initial board configuration."
        
        self.m = m #Number of metafeatures
        self.p = len(self.PRIMITIVES['PREPROCESSING'].values()) + 1 #Length of pipeline
        self.o = len(self.OPERATIONS)
        # Create the empty board array.
        self.pieces_m = [0] * self.m
        self.pieces_p = [0] * self.p
        self.pieces_p = pipeline
        self.pieces_o = [0] * self.o
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

    def findWin(self, player, eval_val=None):
        """Find win of the given color in row, column, or diagonal
        (1 for x, -1 for o)"""
        #print(self[0:])
        if not any(self[0:]):
            return False
        if eval_val == float('inf'):
            return False

        return eval_val >= self.win_threshold

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
