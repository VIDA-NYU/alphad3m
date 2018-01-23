from Coach import Coach
from pipeline.PipelineGame import PipelineGame
from pipeline.pytorch.NNet import NNetWrapper as nn



class PipelineGenerator:

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
        'load_folder_file': ('/dev/models/8x100x50','best.pth.tar')
    })
    
    def createPipelines(self, dataset_path, problem_path):
        self.args['dataset_path'] = dataset_path
        self.args['problem_path'] = problem_path
        self.args['metric'] = problem_path        
        g = PipelineGame(3, self.args)
        nnet = nn(g)
        
        if self.args.get('load_model'):
            nnet.load_checkpoint(self.args.get('load_folder_file')[0], self.args.get('load_folder_file')[1])
            
        c = Coach(g, nnet, self.args)
        c.learn()

    
if __name__=="__main__":
    pg = PipelineGenerator()
    dataset_path = '/Users/yamuna/D3M/data/LL0/LL0_21_car/LL0_21_car_dataset'
    problem_path = '/Users/yamuna/D3M/data/LL0/LL0_21_car/LL0_21_car_problem'
    pg.createPipelines(dataset_path, problem_path)
