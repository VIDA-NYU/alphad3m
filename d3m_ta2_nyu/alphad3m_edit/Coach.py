import logging
from collections import deque
from .Arena import Arena
from .MCTS import MCTS
import numpy as np
from .pytorch_classification.utils import Bar, AverageMeter
import time

logger = logging.getLogger(__name__)

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.board = game.getInitBoard()
        self.nnet = nnet
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        self.board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        
        while episodeStep <= 100:
            self.game.display(self.board)
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(self.board,self.curPlayer)
            temp = int(episodeStep < self.args.get('tempThreshold'))

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getTrainExamples(canonicalBoard, pi)

            trainExamples.append(sym)

            for i in range(self.game.m+self.game.p+self.game.o, len(canonicalBoard)):
                canonicalBoard[i] = 0
            valids = self.game.getValidMoves(canonicalBoard, 1, True)
            logger.info("%s", valids)
            logger.info("%s", pi)
            pi = pi*valids

            if np.sum(pi) == 0:
                break

            if np.sum(pi) != 1:
                pi /= np.sum(pi)
            action = np.random.choice(len(pi), p=pi)

            self.board, self.curPlayer = self.game.getNextState(self.board, self.curPlayer, action)
            r = self.game.getGameEnded(self.board, self.curPlayer)
            if r!=0:
               break
        return trainExamples
    
    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        trainExamples = deque([], maxlen=self.args.get('maxlenOfQueue'))
        for i in range(self.args.get('numIters')):
            # bookkeeping
            logger.info('------ITER %d------', i + 1)
            eps_time = AverageMeter()
            bar = Bar('Self Play', max=self.args.get('numEps'))
            end = time.time()

            for eps in range(self.args.get('numEps')):
                self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree
                trainExamples += self.executeEpisode()



                # bookkeeping + plot progress
                eps_time.update(time.time() - end)
                end = time.time()
                bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.get('numEps'), et=eps_time.avg,
                                                                                                           total=bar.elapsed_td, eta=bar.eta_td)
                bar.next()
            bar.finish()
            
            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.get('checkpoint'), filename='temp.pth.tar')
            pnet = self.nnet.__class__(self.game)
            pnet.load_checkpoint(folder=self.args.get('checkpoint'), filename='temp.pth.tar')
            pmcts = MCTS(self.game, pnet, self.args)
            boards, pis, vs = list(zip(*trainExamples))
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            logger.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game, self.game.display)
            pwins, nwins = arena.playGames(self.args.get('arenaCompare'))

            logger.info('EVALUATIONS %s', self.game.evaluations)
            logger.info('NEW/PREV WINS : %s/%s', nwins, pwins)
            if float(nwins)/(pwins+nwins) < self.args['updateThreshold']:
                logger.info('REJECTING NEW MODEL')
                self.nnet = pnet

            else:
                logger.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename='checkpoint_' + str(i) + '.pth.tar')
                self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename='best.pth.tar')