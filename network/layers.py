# The attention mechanism in this code, along with a small other part, is referenced from:
# Cui Y, Knoll A. Dual-domain strip attention for image restoration. Neural Networks 2024;171:429–39.
# https://doi.org/10.1016/j.neunet.2023.12.003.
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel,
                                             kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


class EncoderBlock(nn.Module):
    def __init__(self, channel):
        super(EncoderBlock, self).__init__()
        self.conv1_1 = BasicConv(channel, channel*2, kernel_size=3, stride=1, norm=False, relu=True)
        self.conv1_2 = BasicConv(channel*2, channel, kernel_size=3, stride=1, norm=False, relu=True)
        self.conv2 = BasicConv(channel, channel, kernel_size=3, stride=1, norm=False, relu=False)
        self.cubic_7 = cubic_attention(channel // 2, group=1, kernel=7)
        self.cubic_13 = cubic_attention(channel // 2, group=1, kernel=13)
        self.global_att = GlobalPoolStripAttention(channel)
        self.norm1 = nn.BatchNorm2d(channel)

    def forward(self, x):
        sp = x
        sp = self.conv1_1(sp)
        sp = self.conv1_2(sp)
        sp = self.global_att(sp)
        sp = torch.chunk(sp, 2, dim=1)
        sp_7 = self.cubic_7(sp[0])
        sp_13 = self.cubic_13(sp[1])
        sp_all = torch.cat((sp_7, sp_13), dim=1)
        sp_all = self.conv2(sp_all)
        output = self.norm1(sp_all + x)
        return output


class DecoderBlock(nn.Module):
    def __init__(self, channel, bridgeflag: bool = False):
        super(DecoderBlock, self).__init__()
        self.bridgeflag = bridgeflag
        self.cubic_7 = cubic_attention(channel // 2, group=1, kernel=7)
        self.cubic_13 = cubic_attention(channel // 2, group=1, kernel=13)
        self.global_att = GlobalPoolStripAttention(channel)
        self.norm1 = nn.BatchNorm2d(channel)

        if self.bridgeflag：
            self.conv1_1 = BasicConv(channel, channel*2, kernel_size=3, stride=1, norm=False, relu=True)
            self.conv1_2 = BasicConv(channel*2, channel, kernel_size=3, stride=1, norm=False, relu=False)
            self.conv2_1 = BasicConv(channel, channel, kernel_size=3, stride=1, norm=False, relu=True)
        else:
            self.conv_en1 = BasicConv(channel, channel*2, kernel_size=3, stride=1, norm=False, relu=True)
            self.conv_de1 = BasicConv(channel, channel*2, kernel_size=3, stride=1, norm=False, relu=True)
            self.conv_en2 = BasicConv(channel*2, channel, kernel_size=3, stride=1, norm=False, relu=False)
            self.conv_de2 = BasicConv(channel*2, channel, kernel_size=3, stride=1, norm=False, relu=False)
            self.merge = BasicConv(channel * 2, channel, kernel_size=3, stride=1, norm=False, relu=False)

    def forward(self, x):
        if self.bridgeflag:
            x1 = self.conv1_1(x)
            x1 = self.conv1_2(x1)
            x1 = self.global_att(x1)
            x1 = torch.chunk(x1, 2, dim=1)
            d_7 = self.cubic_7(x1[0])
            d_13 = self.cubic_13(x1[1])
            d_e_all = torch.cat((d_7, d_13), dim=1)
            weight_matrix = self.conv2_1(d_e_all)
            output = self.norm1(x + weight_matrix)
        else:
            en, de = x
            en_conv = self.conv_en1(en)
            en_conv = self.conv_en2(en_conv)
            en_conv = self.global_att(en_conv)
            de_conv = self.conv_de1(de)
            de_conv = self.conv_de2(de_conv)
            de_conv = self.global_att(de_conv)
            en_conv = torch.chunk(en_conv, 2, dim=1)
            de_conv = torch.chunk(de_conv, 2, dim=1)
            en_7 = self.cubic_7(en_conv[0])
            en_13 = self.cubic_13(en_conv[1])
            en_all = torch.cat((en_7, en_13), dim=1)
            de_7 = self.cubic_7(de_conv[0])
            de_13 = self.cubic_13(de_conv[1])
            de_all = torch.cat((de_7, de_13), dim=1)
            decoder_all = torch.cat((en_all, de_all), dim=1)
            decoder_all = self.merge(decoder_all)
            output = self.norm1(de + decoder_all)
        return output


class DecoderBlock2(nn.Module):
    def __init__(self, channel):
        super(DecoderBlock2, self).__init__()
        self.conv1 = BasicConv(channel, channel, kernel_size=3, stride=1, norm=False, relu=True)
        self.norm1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        d_conv = self.conv1(x)
        output = self.relu(self.norm1(d_conv + x))
        return output


class cubic_attention(nn.Module):
    def __init__(self, dim, group, kernel) -> None:
        super().__init__()
        self.H_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta


class spatial_strip_att(nn.Module):
    def __init__(self, dim, kernel=5, group=2, H=True) -> None:
        super().__init__()
        self.k = kernel
        pad = kernel // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel//2, 1) if H else (1, kernel//2)
        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group*kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Sigmoid()

    def forward(self, x):
        filter = self.ap(x)
        filter = self.conv(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel).reshape(n, self.group, c//self.group, self.k, h*w)
        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1//self.k, self.k, p*q).unsqueeze(2)
        filter = self.filter_act(filter)
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
        return out


class GlobalPoolStripAttention(nn.Module):
    def __init__(self, k) -> None:
        super().__init__()
        self.channel = k
        self.vert_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.vert_high = nn.Parameter(torch.zeros(k, 1, 1))
        self.hori_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.hori_high = nn.Parameter(torch.zeros(k, 1, 1))
        self.vert_pool = nn.AdaptiveAvgPool2d((1, None))
        self.hori_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.gamma = nn.Parameter(torch.zeros(k, 1, 1))
        self.beta = nn.Parameter(torch.ones(k, 1, 1))

    def forward(self, x):
        hori_l = self.hori_pool(x)
        hori_h = x - hori_l
        hori_out = self.hori_low * hori_l + (self.hori_high + 1.) * hori_h
        vert_l = self.vert_pool(hori_out)
        vert_h = hori_out - vert_l
        vert_out = self.vert_low * vert_l + (self.vert_high + 1.) * vert_h
        return x * self.beta + vert_out * self.gamma
