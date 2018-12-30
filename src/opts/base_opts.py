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
        self.parser.add_argument('-DEBUG', type=int, default=0, help='Debug')

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