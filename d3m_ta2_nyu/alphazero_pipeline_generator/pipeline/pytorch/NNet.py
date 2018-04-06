import numpy as np
import os
import time
import torch
from torch.autograd import Variable
import torch.optim as optim

from .PipelineNNet import PipelineNNet as onnet
from ...pytorch_classification.utils import Bar, AverageMeter
from ...NeuralNet import NeuralNet


args = dict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 2,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.get('cuda'):
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args.get('epochs')):
            print('EPOCH ::: ' + str(epoch+1))
            self.nnet.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            batch_size = args.get('batch_size')
            bar = Bar('Training Net', max=int(len(examples)/batch_size))
            batch_idx = 0

            while batch_idx < int(len(examples)/batch_size):
                sample_ids = np.random.randint(len(examples), size=batch_size)
                #print('SAMPLE IDS ', sample_ids, ' ', examples[0])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = [board[0:self.board_x] for board in boards]
                #print('\n\nBOARDS\n', boards)
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.get('cuda'):
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
                else:
                    boards, target_pis, target_vs = Variable(boards), Variable(target_pis), Variable(target_vs)

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.data[0], boards.size(0))
                v_losses.update(l_v.data[0], boards.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                            batch=batch_idx,
                            size=int(len(examples)/batch_size),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            lpi=pi_losses.avg,
                            lv=v_losses.avg,
                            )
                bar.next()
            bar.finish()


    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()
        print('\n\n\nBOARD\n', [r[1] for r in board[0:self.board_x]])

        # preparing input
        board = torch.from_numpy(np.array(board[0:self.board_x], dtype='f')).cuda().double() if args.get('cuda') else torch.from_numpy(np.array(board[0:self.board_x], dtype='f'))
        #board = torch.FloatTensor(board[0:self.board_x])
        if args.get('cuda'): board = board.contiguous().cuda()
        board = Variable(board, volatile=True)
        board = board.view(1, self.board_x, self.board_y)

        self.nnet.eval()
        pi, v = self.nnet(board)

        print('\n\n\nPROBABILITY\n', torch.exp(pi).data.cpu().numpy()[0])
        print('\n\n\nVALUE\n',  v.data.cpu().numpy()[0])

        print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets*outputs)/targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict' : self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(checkpoint))
        checkpoint = torch.load(filepath)
        self.nnet.load_state_dict(checkpoint['state_dict'])