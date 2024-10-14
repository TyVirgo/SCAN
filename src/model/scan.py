import torch
import torch.nn as nn
from . import block as B


def make_model(args, parent=False):
    model = SCAN(upscale=args.scale[0])
    return model


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class SCAN(nn.Module):
    def __init__(self, in_nc=3, nf=48, num_modules=4, out_nc=3, upscale=4):
        super(SCAN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        self.sub_mean = MeanShift(1)
        kel_nels = [7, 7, 3, 3]
        self.B1 = B.DRAB(in_channels=nf, kel_nel=kel_nels[0])
        self.eca1 = B.RKAB(nf, kel_nels[0])
        self.B2 = B.DRAB(in_channels=nf, kel_nel=kel_nels[1])
        self.eca2 = B.RKAB(nf, kel_nels[1])
        self.B3 = B.DRAB(in_channels=nf, kel_nel=kel_nels[2])
        self.eca3 = B.RKAB(nf, kel_nels[2])
        self.B4 = B.DRAB(in_channels=nf, kel_nel=kel_nels[3])
        self.eca4 = B.RKAB(nf, kel_nels[3])
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.add_mean = MeanShift(1, sign=1)
        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)

    def forward(self, input):
        input = self.sub_mean(input)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_1 = self.eca1(out_B1)

        out_B2 = self.B2(out_B1)
        out_2 = self.eca2(out_B2)

        out_B3 = self.B3(out_B2)
        out_3 = self.eca3(out_B3)

        out_B4 = self.B4(out_B3)
        out_4 = self.eca4(out_B4)

        out_B = self.c(torch.cat([out_1, out_2, out_3, out_4], dim=1))

        out_lr = out_B + out_fea

        output = self.upsampler(out_lr)

        output = self.add_mean(output)

        return output
