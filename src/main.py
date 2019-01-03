import numpy as np
import torch
import random
from opts.base_opts import Opts
from data_process.data_loader_provider import create_data_loaders
from model.model_provider import create_model, create_optimizer
from training.train_net import train_net, validate_net


def main():
    # Seed all sources of randomness to 0 for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)

    opt = Opts().parse()

    # Create data loaders
    train_loader, test_loader = create_data_loaders(opt)

    # Create nn
    model, criterion_hm, criterion_paf = create_model(opt)
    model = model.cuda()
    criterion_hm = criterion_hm.cuda()
    criterion_paf = criterion_paf.cuda()

    # Create optimizer
    optimizer = create_optimizer(opt, model)

    # Other params
    n_epochs = opt.nEpoch
    to_train = opt.train
    drop_lr = opt.dropLR
    val_interval = opt.valInterval
    learn_rate = opt.LR
    visualize_out = opt.vizOut

    # train/ test
    if to_train:
        train_net(train_loader, test_loader, model, criterion_hm, criterion_paf, optimizer, n_epochs,
                  val_interval, learn_rate, drop_lr, opt.saveDir, visualize_out)
    else:
        validate_net(test_loader, model, criterion_hm, criterion_paf, viz_output=visualize_out)


if __name__ == '__main__':
    main()
