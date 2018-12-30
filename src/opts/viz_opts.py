from opts.base_opts import Opts


class VizOpts(Opts):
    def __init__(self):
        super().__init__()

    def init(self):
        super().init()
        self.parser.add_argument('-vizIgnoreMask', dest='vizIgnoreMask', action='store_true', help='Visualize Ignore Mask')
        self.parser.add_argument('-vizHeatMap', dest='vizHeatMap', action='store_true', help='Visualize Heatmap')
        self.parser.add_argument('-vizPaf', dest='vizPaf', action='store_true', help='Visualize PAF')

