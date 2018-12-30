import torch.nn as nn
from .helper import init, make_standard_block
import torch


class PAFModel(nn.Module):
    def __init__(self, backend, backend_outp_feats, n_joints, n_paf, n_stages=7):
        super(PAFModel, self).__init__()
        assert (n_stages > 0)
        self.backend = backend
        stages = [Stage(backend_outp_feats, n_joints, n_paf, True)]
        for i in range(n_stages - 1):
            stages.append(Stage(backend_outp_feats, n_joints, n_paf, False))
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        img_feats = self.backend(x)
        cur_feats = img_feats
        heatmap_outs = []
        paf_outs = []
        for i, stage in enumerate(self.stages):
            heatmap_out, paf_out = stage(cur_feats)
            heatmap_outs.append(heatmap_out)
            paf_outs.append(paf_out)
            cur_feats = torch.cat([img_feats, heatmap_out, paf_out], 1)
        return heatmap_outs, paf_outs


class Stage(nn.Module):
    def __init__(self, backend_outp_feats, n_joints, n_paf, stage1):
        super(Stage, self).__init__()
        inp_feats = backend_outp_feats
        if stage1:
            self.block1 = make_paf_block_stage1(inp_feats, n_joints)
            self.block2 = make_paf_block_stage1(inp_feats, n_paf)
        else:
            inp_feats = backend_outp_feats + n_joints + n_paf
            self.block1 = make_paf_block_stage2(inp_feats, n_joints)
            self.block2 = make_paf_block_stage2(inp_feats, n_paf)
        init(self.block1)
        init(self.block2)

    def forward(self, x):
        y1 = self.block1(x)
        y2 = self.block2(x)
        return y1, y2


def make_paf_block_stage1(inp_feats, output_feats):
    layers = [make_standard_block(inp_feats, 128, 3),
              make_standard_block(128, 128, 3),
              make_standard_block(128, 128, 3),
              make_standard_block(128, 512, 1, 1, 0)]
    layers += [nn.Conv2d(512, output_feats, 1, 1, 0)]
    return nn.Sequential(*layers)


def make_paf_block_stage2(inp_feats, output_feats):
    layers = [make_standard_block(inp_feats, 128, 7, 1, 3),
              make_standard_block(128, 128, 7, 1, 3),
              make_standard_block(128, 128, 7, 1, 3),
              make_standard_block(128, 128, 7, 1, 3),
              make_standard_block(128, 128, 7, 1, 3),
              make_standard_block(128, 128, 1, 1, 0)]
    layers += [nn.Conv2d(128, output_feats, 1, 1, 0)]
    return nn.Sequential(*layers)
