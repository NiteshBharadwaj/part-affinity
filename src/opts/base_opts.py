import argparse
import os

class Opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('-expID', default='default', help='Experiment ID')
        self.parser.add_argument('-data', default='default', help='Input data folder')
        self.parser.add_argument('-nThreads', default=4, type=int, help='Number of threads')
        self.parser.add_argument('-expDir', default='../exp', help='Experiments directory')
        self.parser.add_argument('-scaleAugFactor', default=0.25, type=float, help='Scale augment factor')
        self.parser.add_argument('-rotAugProb', default=0.4, type=float, help='Rotation augment probability')
        self.parser.add_argument('-flipAugProb', default=0.5, type=float, help='Flip augment probability')
        self.parser.add_argument('-rotAugFactor', default=30, type=float, help='Rotation augment factor')
        self.parser.add_argument('-colorAugFactor', default=0.2, type=float, help='Colo augment factor')
        self.parser.add_argument('-imgSize', default=368, type=int, help='Number of threads')
        self.parser.add_argument('-hmSize', default=46, type=int, help='Number of threads')
        self.parser.add_argument('-DEBUG', type=int, default=0, help='Debug')
        self.parser.add_argument('-sigmaPAF', default=5, type=int, help='Width of PAF')
        self.parser.add_argument('-sigmaHM', default=1, type=int, help='Std. of Heatmap')
        self.parser.add_argument('-variableWidthPAF', dest='variableWidthPAF', action='store_true', help='Variable width PAF based on length of part')
        self.parser.add_argument('-dataset', default='coco', help='Dataset')
        self.parser.add_argument('-model', default='vgg', help='Model')
        self.parser.add_argument('-batchSize', default=8, type=int, help='Batch Size')
        self.parser.add_argument('-LR', default=1e-3, type=float, help='Learn Rate')
        self.parser.add_argument('-nEpoch', default=20, type=int, help='Number of Epochs')
        self.parser.add_argument('-dropLR', type=float, default=10, help='Drop LR')
        self.parser.add_argument('-valInterval', type=int, default=1, help='Val Interval')
        self.parser.add_argument('-loadModel', default='none', help='Load pre-trained')
        self.parser.add_argument('-train', dest='train', action='store_true', help='Train')
        self.parser.add_argument('-vizOut', dest='vizOut', action='store_true', help='Visualize output?')
        self.parser.add_argument('-criterionHm', default='mse', help='Heatmap Criterion')
        self.parser.add_argument('-criterionPaf', default='mse', help='PAF Criterion')

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()
        self.opt.saveDir = os.path.join(self.opt.expDir, self.opt.expID)
        if self.opt.DEBUG > 0:
            self.opt.nThreads = 1

        args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))
        if not os.path.exists(self.opt.saveDir):
            os.makedirs(self.opt.saveDir)
        file_name = os.path.join(self.opt.saveDir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
        return self.opt