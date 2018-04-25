import sys
#sys.path.append('..')
from ...utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class PipelineNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.action_size = game.getActionSize()
        self.board_size = game.getBoardSize()
        #self.op_size = game.getOperationsSize()
        self.args = args

        super(PipelineNNet, self).__init__()

        self.lstm = nn.LSTM(self.board_size, 512, 2)
        #print('ACTION SIZE ', self.action_size)
        self.probFC = nn.Linear(512, self.action_size)
        self.valueFC = nn.Linear(512, 1)

    def forward(self, s):
        s = s.view(-1, 1, self.board_size)
        lstm_out, hidden = self.lstm(s)
        s = lstm_out[:,-1]
        pi = self.probFC(s)                                                                         # batch_size x 512
        v = self.valueFC(s)                                                                          # batch_size x 512
        return F.log_softmax(pi, 1), F.sigmoid(v)
