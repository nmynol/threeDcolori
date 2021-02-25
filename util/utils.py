import os

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from data.dataloader import MyDataset


def create_dataloader(opt):
    dataset = MyDataset(opt.img_path, opt.xdog_path, opt.img_size, opt.seq_len, opt.threshold)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                             pin_memory=True, num_workers=opt.workers)
    return data_loader


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_state(epoch, epoch_iter, total_iter, dict, writer):
    message = '[epoch: %d, iters: %d] ' % (epoch, epoch_iter)
    for key in dict:
        message += '%s: %.3f ' % (key, dict[key])
        writer.add_scalar(str(key), dict[key], total_iter)
    print(message)


def display_pic(visuals, total_iter, opt, writer):
    xdog, img, gt = visuals
    for i in range(img.shape[0]):
        # print(opti.shape)
        # print('saving ' + str(data_type) + ' sample...')
        add_image(xdog[i], img[i], gt[i], i, opt.seq_len, opt.img_size, total_iter, writer)


def add_image(xdog, opti, gt, i, seq_len, img_size, total_iter, writer):
    xdog_frames = cat_xdog(xdog, seq_len, img_size)
    opti_frames = cat_image(opti, seq_len, img_size)
    gt_frames = cat_image(gt, seq_len, img_size)
    cluster = torch.zeros((3, gt_frames.shape[1] * 3, gt_frames.shape[2]))
    cluster[:, 0:gt_frames.shape[1], :] = xdog_frames
    cluster[:, gt_frames.shape[1]:(gt_frames.shape[1] * 2), :] = opti_frames
    cluster[:, (gt_frames.shape[1] * 2):(gt_frames.shape[1] * 3), :] = gt_frames
    writer.add_image(str(i + 1), torch.clamp(cluster, 0, 255), total_iter)


def cat_image(opti, seq_len, img_size):
    cat = torch.zeros((3, img_size, img_size * seq_len))
    offset = 0
    for i in range(seq_len):
        # print(i)
        # print(cat[:, :, offset:(offset + img_size)].shape)
        # print(opti[i].shape)
        cat[:, :, offset:(offset + img_size)] = opti[i].mul(0.5).add(0.5)
        offset = offset + img_size
    return cat


def cat_xdog(opti, seq_len, img_size):
    cat = torch.zeros((1, img_size, img_size * seq_len))
    offset = 0
    for i in range(seq_len):
        cat[:, :, offset:(offset + img_size)] = opti[i].mul(0.5).add(0.5)
        offset = offset + img_size
    return cat