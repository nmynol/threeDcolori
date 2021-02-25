import os

import torch
import torch.nn as nn

from models.base_model import BaseModel
from models.networks import Conv3dAttentionBlock
from util.utils import get_scheduler
from util.loss import GANLoss, StyleLoss, PerceptualLoss


class MainModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt

        # set net
        self.netG = torch.nn.DataParallel(Conv3dAttentionBlock())

        if self.isTrain:
            self.netD = torch.nn.DataParallel(Conv3dAttentionBlock())
            # set loss
            self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.styleLoss = StyleLoss().to(self.device)
            self.contentLoss = PerceptualLoss().to(self.device)
            # set optimizer
            self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizerG)
            self.optimizers.append(self.optimizerD)
            # set lr scheduler
            if opt.lr_policy:
                self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def setup(self):
        self.print_networks()
        self.netG.to(self.device)
        self.netD.to(self.device)

    def train(self):
        self.netG.train()
        self.netD.train()

    def eval(self):
        self.netG.eval()
        self.netD.eval()

    def set_input(self, input):
        img_diff, xdog_diff, img_now, img_first, xdog_now, xdog_first = input
        img_diff = img_diff.permute(0, 2, 1, 3, 4)
        xdog_diff = xdog_diff.permute(0, 2, 1, 3, 4)
        img_now = img_now.permute(0, 2, 1, 3, 4)
        xdog_now = xdog_now.permute(0, 2, 1, 3, 4)

        self.img_diff, self.xdog_diff = img_diff.to(self.device), xdog_diff.to(self.device)
        self.img_now, self.img_first = img_now.to(self.device), img_first.to(self.device)
        self.xdog_now, self.xdog_first = xdog_now.to(self.device), xdog_first.to(self.device)

    def optimize_parameters(self):
        self.forward()
        # update D
        # self.set_requires_grad(self.netD, True)
        # self.set_requires_grad(self.netG, False)
        # self.optimizerD.zero_grad()
        # self.backward_D()
        # self.optimizerD.step()
        # update G
        self.set_requires_grad(self.netG, True)
        # self.set_requires_grad(self.netD, False)
        self.optimizerG.zero_grad()
        self.backward_G()
        self.optimizerG.step()

    def forward(self):
        self.fake_img = self.netG(self.xdog_now, self.img_diff, self.img_first)

    def backward_G(self):
        # pred_fake = self.netD(self.fake_img, self.img_diff, self.img_first)
        # self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_img, self.img_now) * self.opt.lambda_L1
        self.loss_G_style = self.styleLoss(self.fake_img, self.img_now) * self.opt.lambda_style
        self.loss_G_content = self.contentLoss(self.fake_img, self.img_now) * self.opt.lambda_content

        # self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_style + self.loss_G_content
        self.loss_G = self.loss_G_L1 + self.loss_G_style + self.loss_G_content
        self.loss_G.backward()

    def backward_D(self):
        pred_fake = self.netD(self.fake_img, self.img_diff, self.img_first)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        pred_real = self.netD(self.img_now, self.img_diff, self.img_first)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def test(self):
        with torch.no_grad():
            self.forward()

    def update_learning_rate(self):
        pass

    def get_current_visuals(self):
        return self.xdog_now, self.fake_img, self.img_now

    def get_losses(self):
        return {
            'loss_D': self.loss_D.item(),
            # 'loss_G_GAN': self.loss_G_GAN.item(),
            'loss_G_L1': self.loss_G_L1.item(),
            'loss_G_style': self.loss_G_style.item(),
            'loss_G_content': self.loss_G_content.item()
        }

    def save_networks(self, opt, epoch, total_iter):
        if not os.path.exists(opt.checkpoints_dir):
            os.makedirs(opt.checkpoints_dir)
        state = {
            'state_dictG': self.netG.state_dict(),
            'optimizerG': self.optimizerG.state_dict(),
            'state_dictD': self.netD.state_dict(),
            'optimizerD': self.optimizerD.state_dict(),
            'epoch': epoch,
            'total_iter': total_iter
        }
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iter))
        torch.save(state, os.path.join(opt.checkpoints_dir, 'params_' + str(epoch) + '.pth'))
        torch.save(state, os.path.join(opt.checkpoints_dir, 'params_latest.pth'))
        print("saving completed.")

    def load_networks(self, opt):
        path = opt.continue_train
        assert os.path.isfile(path)
        print("=> loading checkpoint from '{}'".format(path))
        checkpoint = torch.load(path)
        self.netG.load_state_dict(checkpoint['state_dictG'])
        self.optimizerG.load_state_dict(checkpoint['optimizerG'])
        self.netD.load_state_dict(checkpoint['state_dictD'])
        self.optimizerD.load_state_dict(checkpoint['optimizerD'])
        epoch = checkpoint['epoch']
        total_iter = checkpoint['total_iter']
        print("loading completed.")
        return epoch, total_iter

    def print_networks(self):
        print('---------- Networks initialized -------------')
        nets = [self.netG, self.netD]
        num_params = 0
        for net in nets:
            for param in net.parameters():
                num_params += param.numel()
            print(net)
        print('Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')
