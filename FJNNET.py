# The attention mechanism in this code, along with a small other part, is referenced from:
# Cui Y, Knoll A. Dual-domain strip attention for image restoration. Neural Networks 2024;171:429–39.
# https://doi.org/10.1016/j.neunet.2023.12.003.
from layers import *


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(1, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x


class Merge(nn.Module):
    def __init__(self, channel):
        super(Merge, self).__init__()
        self.convmerge = BasicConv(channel * 2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.convmerge(torch.cat([x1, x2], dim=1))


class FJNNet(nn.Module):
    def __init__(self):
        super(FJNNet, self).__init__()
        base_channel = 16

        self.Encoder = nn.ModuleList([
            EncoderBlock(base_channel),
            EncoderBlock(base_channel * 2),
            EncoderBlock(base_channel * 4),
            EncoderBlock(base_channel * 8),
        ])

        self.Decoder = nn.ModuleList([
            DecoderBlock(base_channel * 8, bridgeflag=True),
            DecoderBlock2(base_channel * 4),
            DecoderBlock(base_channel * 4),
            DecoderBlock2(base_channel * 2),
            DecoderBlock(base_channel * 2),
            DecoderBlock2(base_channel),
            DecoderBlock(base_channel),
            DecoderBlock2(base_channel // 2),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(1, base_channel, kernel_size=3, norm=True, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, norm=True, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, norm=True, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 8, kernel_size=3, norm=True, relu=True, stride=2),
            BasicConv(base_channel * 8, base_channel * 4, kernel_size=4, norm=True, relu=True, stride=2,
                      transpose=True),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, norm=True, relu=True, stride=2,
                      transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, norm=True, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, base_channel // 2, kernel_size=3, norm=True, relu=True, stride=1),
            BasicConv(base_channel // 2, 1, kernel_size=3, norm=True, relu=True, stride=1),
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 8, base_channel * 4, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ac_convs = nn.ModuleList([
            BasicConv(1, base_channel, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 4, base_channel * 8, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList([
            BasicConv(base_channel * 4, 1, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, 1, kernel_size=3, relu=False, stride=1),
        ])

        self.merge8 = Merge(base_channel * 8)
        self.SCM8 = SCM(base_channel * 8)
        self.merge4 = Merge(base_channel * 4)
        self.SCM4 = SCM(base_channel * 4)
        self.merge2 = Merge(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        x_8 = F.interpolate(x_4, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM4(x_4)
        z8 = self.SCM8(x_8)

        input5 = list()
        input6 = list()
        input7 = list()

        x1 = self.feat_extract[0](x)
        res1 = self.Encoder[0](x1)

        x2 = self.feat_extract[1](res1)
        x2 = self.merge2(x2, z2)
        res2 = self.Encoder[1](x2)

        x3 = self.feat_extract[2](res2)
        x3 = self.merge4(x3, z4)
        res3 = self.Encoder[2](x3)

        x4 = self.feat_extract[3](res3)
        x4 = self.merge8(x4, z8)
        res4 = self.Encoder[3](x4)

        dres1 = self.Decoder[0](res4)
        d1 = self.feat_extract[4](dres1)
        dres1_1 = self.Decoder[1](d1)

        input5.append(res3)
        input5.append(dres1_1)
        dres2 = self.Decoder[2](input5)
        d2 = self.feat_extract[5](dres2)
        dres2_2 = self.Decoder[3](d2)

        input6.append(res2)
        input6.append(dres2_2)
        dres3 = self.Decoder[4](input6)
        d3 = self.feat_extract[6](dres3)
        dres3_3 = self.Decoder[5](d3)

        input7.append(res1)
        input7.append(dres3_3)
        dres4 = self.Decoder[6](input7)
        d4 = self.feat_extract[7](dres4)
        dres4_4 = self.Decoder[7](d4)
        output = self.feat_extract[8](dres4_4)

        return output
