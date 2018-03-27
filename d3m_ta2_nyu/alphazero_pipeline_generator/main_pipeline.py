from .Coach import Coach
from .pipeline.PipelineGame import PipelineGame
from .pipeline.pytorch.NNet import NNetWrapper as nn


args = dict({
    'numIters': 3,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
})

if __name__=="__main__":
    g = PipelineGame(3)
    nnet = nn(g)

    if args.get('load_model'):
        nnet.load_checkpoint(args.get('load_folder_file')[0], args.get('load_folder_file')[1])

    c = Coach(g, nnet, args)
    c.learn()
