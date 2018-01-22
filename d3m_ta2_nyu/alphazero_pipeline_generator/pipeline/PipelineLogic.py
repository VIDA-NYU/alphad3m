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
            'sklearn.preprocessing.OneHotEncoder': 1
        },
        'IMPUTATION': {
            'sklearn.preprocessing.Imputer': 2
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

    def __init__(self, n=3, problem='CLASSIFICATION'):
        "Set up initial board configuration."

        self.n = n
        # Create the empty board array.
        self.pieces = [None] * self.n
        for i in range(self.n):
            self.pieces[i] = [0] * self.n
        self[0][1] = 1
        self[1][1] = 2
        self.previous_moves = []
        self.win_threshold = 0.8
        self.problem = problem
        
    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces[index]

    def findWin(self, player, eval_val=None):
        """Find win of the given color in row, column, or diagonal
        (1 for x, -1 for o)"""
        #print('IN FINDWIN ', self.pieces, ' ', self[2][0], ' ', eval_val >= self.win_threshold)
        if self[2][1] == 0:
            return False
        return eval_val >= self.win_threshold

    def get_legal_moves(self, problem='CLASSIFICATION'):
        """Returns all the legal moves.
        """
        valid_moves = [0]*(len(self.PRIMITIVES[problem].values())+1)
        count = 0
        #print('PREVIOUS MOVES ', self.previous_moves)
        for primitive in self.PRIMITIVES[problem].values():
            if primitive not in self.previous_moves:
                valid_moves[count] = 1
            count += 1
        if np.sum(valid_moves) == 0:
            valid_moves[-1] = 1
        return valid_moves
        
    def has_legal_moves(self, problem='CLASSIFICATION'):
        return len(self.previous_moves) < len(self.PRIMITIVES[problem].values())

    def getMove(self, action):
        return [val for val in Board.PRIMITIVES[self.problem].values()][action]
    
    def execute_move(self, move, player):
        """Perform the given move on the board;
        color gives the color of the piece to play (1=x,-1=o)
        """

        # Add the piece to the empty square.
        # print(move)
        # print(self[x][y],color)
        x = 2
        y = 1
        self[x][y] = move
        if not move in self.previous_moves:
            self.previous_moves.append(move)

