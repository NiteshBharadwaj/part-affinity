import torch
import os
from tqdm import tqdm
from .eval import eval
from visualization.visualize import visualize_output


def step(data_loader, model, criterion_hm, criterion_paf, to_train=False, optimizer=None, viz_output=False):
    if to_train:
        model.train()
    else:
        model.eval()
    nIters = len(data_loader)
    hm_loss_meter, paf_loss_meter = AverageMeter(), AverageMeter()
    outputs = []
    with tqdm(total=nIters) as t:
        for i, (input_, heatmap, paf, ignore_mask, indices) in enumerate(data_loader):
            input_cuda = input_.float().cuda()
            heatmap_t_cuda = heatmap.float().cuda()
            paf_t_cuda = paf.float().cuda()
            ignore_mask_cuda = ignore_mask.reshape(ignore_mask.shape[0], 1,
                                                   ignore_mask.shape[1], ignore_mask.shape[2]).float().cuda()
            allow_mask = 1 - ignore_mask_cuda
            heatmap_outputs, paf_outputs = model(input_cuda)
            loss_hm_total = 0
            loss_paf_total = 0
            for i in range(len(heatmap_outputs)):
                heatmap_out = heatmap_outputs[i]
                paf_out = paf_outputs[i]
                loss_hm_total += criterion_hm(heatmap_out * allow_mask, heatmap_t_cuda * allow_mask)/allow_mask.sum().detach()/heatmap.shape[0]/heatmap.shape[1]
                loss_paf_total += criterion_paf(paf_out * allow_mask, paf_t_cuda * allow_mask)/allow_mask.sum().detach()/heatmap.shape[0]/paf.shape[1]
            loss = loss_hm_total + loss_paf_total
            output = (heatmap_outputs[-1].data.cpu().numpy(), paf_outputs[-1].data.cpu().numpy(), indices.numpy())
            if to_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                outputs.append(output)
            if viz_output:
                visualize_output(input_, heatmap, paf, ignore_mask, output)
            hm_loss_meter.update(loss_hm_total.data.cpu().numpy())
            paf_loss_meter.update(loss_paf_total.data.cpu().numpy())
            t.set_postfix(loss_hm='{:05.3f}'.format(hm_loss_meter.avg), loss_paf='{:05.3f}'.format(paf_loss_meter.avg))
            t.update()
    return hm_loss_meter.avg, paf_loss_meter.avg, outputs


def train_net(train_loader, test_loader, model, criterion_hm, criterion_paf, optimizer,
              n_epochs, val_interval, learn_rate, drop_lr, save_dir, viz_output=False):
    heatmap_loss_avg, paf_loss_avg = 0.0, 0.0
    for epoch in range(1, n_epochs + 1):
        step(train_loader, model, criterion_hm, criterion_paf, True, optimizer, viz_output=viz_output)
        if epoch % val_interval == 0:
            test_net(test_loader, model, criterion_hm, criterion_paf, save_dir, epoch, viz_output=viz_output)
        adjust_learning_rate(optimizer, epoch, drop_lr, learn_rate)
    return heatmap_loss_avg, paf_loss_avg


def test_net(test_loader, model, criterion_hm, criterion_paf, save_dir=None, epoch=0, viz_output=False):
    heatmap_loss_avg, paf_loss_avg, outputs = step(test_loader, model, criterion_hm, criterion_paf, viz_output=viz_output)
    if not save_dir is None:
        torch.save(model, os.path.join(save_dir, 'model_{}.pth'.format(epoch)))
    eval(test_loader, outputs)
    return heatmap_loss_avg, paf_loss_avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, dropLR, LR):
    lr = LR * (0.1 ** (epoch // dropLR))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
