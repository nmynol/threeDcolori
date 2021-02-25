import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskNet(nn.Module):
    def __init__(self):
        super(MaskNet, self).__init__()
        self.fc = nn.Linear(300, 400)

    def forward(self, x):
        return x


class Conv3dAttentionBlock(nn.Module):
    def __init__(self):
        super(Conv3dAttentionBlock, self).__init__()

    def forward(self, x, y, z):  # (batch_size, channel, seq_len, img_size, img_size)
        x = torch.cat((x, y, torch.cat([torch.unsqueeze(z, 2) for i in range(8)], dim=2)), dim=1)
        print('x:', x.shape)
        q = F.conv3d(x, nn.Parameter(torch.randn(x.shape[1], x.shape[1], 1, 1, 1)))
        print('q:', q.shape)
        q = F.relu(q)
        k = F.conv3d(x, nn.Parameter(torch.randn(x.shape[1], x.shape[1], 1, 1, 1)))
        print('k:', k.shape)
        k = F.relu(k)
        v = F.conv3d(x, nn.Parameter(torch.randn(x.shape[0], x.shape[1], 1, 1, 1)))
        print('v:', v.shape)
        v = F.relu(v)
        q_k = F.conv3d(q, k)
        print('q_k:', q_k.shape)
        q_k = F.relu(q_k)
        h = F.conv3d(v, q_k)
        print('h:', h.shape)
        h = F.relu(h)
        h = F.conv3d(h, nn.Parameter(torch.randn(3, h.shape[0], 1, 1, 1)))
        return h


if __name__ == '__main__':
    x = torch.randn(2, 1, 8, 224, 224)
    y = torch.randn(2, 1, 8, 224, 224)
    z = torch.randn(2, 3, 224, 224)
    block = Conv3dAttentionBlock()
    print(block(x, y, z).shape)
    sum = 0
    for param in block.parameters():
        sum += param.numel()
    print(sum)
