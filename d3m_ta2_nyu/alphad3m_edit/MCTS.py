import math
import numpy as np

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vals = {}
        self.Vs = {}       # stores game.getValidMoves for board s
        self.count = 0
        self.prev_moves = []

    def getActionProb(self, canonicalBoard, temp=1, problem='CLASSIFICATION'):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.get('numMCTSSims')):
            self.search(canonicalBoard, problem=problem)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            if np.sum(probs) == 0:
                print('PROB ZEROz')
            return probs

        counts = [x**(1./temp) for x in counts]
        if np.sum(counts) == 0:
            probs = [1/(len(counts))]*len(counts)
        else:
            probs = [x/float(sum(counts)) for x in counts]
        if np.sum(probs) == 0:
            print('PROB ZERO')
        return probs


    def search(self, canonicalBoard, player=1, problem='CLASSIFICATION'):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        
        s = self.game.stringRepresentation(canonicalBoard)
        game_ended = self.game.getGameEnded(canonicalBoard, player)
        if not s in self.Es:
            self.Es[s] = game_ended
        elif self.Es[s] != game_ended:
            self.Es[s] = game_ended
        if self.Es[s]!=0:
            # terminal node
            #Clear all previous moves
            self.prev_moves = []
            return self.Vals[s] if not self.Vals.get(s) is None else 0

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            print('Value ', v)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            self.Ps[s] /= np.sum(self.Ps[s])    # renormalize
            self.Vals[s] = v
            self.Vs[s] = valids
            self.Ns[s] = 0
            return v

        valids = self.Vs[s]

        #Check if valid moves are available. Quit if no more legal moves are possible
        #if not any(valids):
        #    return -1
        
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args.get('cpuct')*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.args.get('cpuct')*self.Ps[s][a]*math.sqrt(self.Ns[s])     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        if a in self.prev_moves:
            return 0

        #Append action to previous moves
        self.prev_moves.append(a)

        next_s, next_player = self.game.getNextState(canonicalBoard, player, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        #print('NEXT STATE SEARCH RECURSION')
        v = self.search(next_s, next_player, problem)

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return v
