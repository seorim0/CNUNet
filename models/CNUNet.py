import torch
import torch.nn as nn
import torch.nn.functional as functional
from utils import ConvSTFT, ConviSTFT, complex_cat


# causal convolution
class causalConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=1, dilation=1, groups=1):
        super(causalConv2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=(padding[0], 0),
                              dilation=dilation, groups=groups)
        self.padding = padding[1]

    def forward(self, x):
        x = functional.pad(x, [self.padding, 0, 0, 0])
        out = self.conv(x)
        return out


# convolution block
class CONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CONV, self).__init__()
        self.conv = causalConv2d(in_ch, out_ch, kernel_size=(3, 2), stride=(2, 1), padding=(1, 1))
        self.ln = nn.GroupNorm(1, out_ch, eps=1e-8)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.ln(self.conv(x)))


# convolution block for input layer
class INCONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(INCONV, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.ln = nn.GroupNorm(1, out_ch, eps=1e-8)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.ln(self.conv(x)))


# sub-pixel convolution block
class SPCONV(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(SPCONV, self).__init__()
        self.conv = causalConv2d(in_ch, out_ch * scale_factor, kernel_size=(3, 2), padding=(1, 1))
        self.ln = nn.GroupNorm(1, out_ch, eps=1e-8)
        self.prelu = nn.PReLU()

        self.n = scale_factor

    def forward(self, x):
        x = self.conv(x)  # [B, C, F, T]

        x = x.permute(0, 3, 2, 1)  # [B, T, F, C]
        r = torch.reshape(x, (x.size(0), x.size(1), x.size(2), x.size(3) // self.n, self.n))  # [B, T, F, C//2 , 2]
        r = r.permute(0, 1, 2, 4, 3)  # [B, T, F, 2, C//2]
        r = torch.reshape(r, (x.size(0), x.size(1), x.size(2) * self.n, x.size(3) // self.n))  # [B, T, F*2, C//2]
        r = r.permute(0, 3, 2, 1)  # [B, C, F, T]

        out = self.ln(r)
        out = self.prelu(out)
        return out


# sub-pixel convolution block with complex computation
class CSPCONV(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(CSPCONV, self).__init__()
        self.conv_r = causalConv2d(in_ch, out_ch * scale_factor, kernel_size=(3, 2), padding=(1, 1))
        self.conv_i = causalConv2d(in_ch, out_ch * scale_factor, kernel_size=(3, 2), padding=(1, 1))

        self.ln_r = nn.GroupNorm(1, out_ch, eps=1e-8)
        self.prelu_r = nn.PReLU()
        self.ln_i = nn.GroupNorm(1, out_ch, eps=1e-8)
        self.prelu_i = nn.PReLU()

        self.n = scale_factor

    def forward(self, real_x, imag_x):
        real2real = self.conv_r(real_x)  # [B, C, F, T]
        real2imag = self.conv_i(real_x)
        imag2imag = self.conv_i(imag_x)
        imag2real = self.conv_r(imag_x)

        real_x = real2real - imag2imag
        imag_x = real2imag + imag2real

        # real spc
        real_x = real_x.permute(0, 3, 2, 1)  # [B, T, F, C]
        r = torch.reshape(real_x, (
            real_x.size(0), real_x.size(1), real_x.size(2), real_x.size(3) // self.n, self.n))  # [B, T, F, C//2 , 2]
        r = r.permute(0, 1, 2, 4, 3)  # [B, T, F, 2, C//2]
        r = torch.reshape(r, (
            real_x.size(0), real_x.size(1), real_x.size(2) * self.n, real_x.size(3) // self.n))  # [B, T, F*2, C//2]
        r = r.permute(0, 3, 2, 1)  # [B, C, F, T]

        # imag spc
        imag_x = imag_x.permute(0, 3, 2, 1)  # [B, T, F, C]
        i = torch.reshape(imag_x, (
            imag_x.size(0), imag_x.size(1), imag_x.size(2), imag_x.size(3) // self.n, self.n))  # [B, T, F, C//2 , 2]
        i = i.permute(0, 1, 2, 4, 3)  # [B, T, F, 2, C//2]
        i = torch.reshape(i, (
            imag_x.size(0), imag_x.size(1), imag_x.size(2) * self.n, imag_x.size(3) // self.n))  # [B, T, F*2, C//2]
        i = i.permute(0, 3, 2, 1)  # [B, C, F, T]

        out_r = self.ln_r(r)
        out_r = self.prelu_r(out_r)

        out_i = self.ln_i(i)
        out_i = self.prelu_i(out_i)
        return out_r, out_i


# 1x1 conv for down-sampling
class down_sampling_half(nn.Module):
    def __init__(self, in_ch):
        super(down_sampling_half, self).__init__()
        self.down_sampling1 = nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.down_sampling2 = nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

    def forward(self, x):
        _, C, _, _ = x.size()
        out1 = self.down_sampling1(x[:, :C // 2])
        out2 = self.down_sampling2(x[:, C // 2:])
        return torch.cat([out1, out2], dim=1)


# 1x1 conv for up-sampling
class upsampling_half(nn.Module):
    def __init__(self, in_ch):
        super(upsampling_half, self).__init__()
        self.upsampling1 = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=(3, 1), stride=(2, 1),
                                              padding=(1, 0), output_padding=(1, 0))
        self.upsampling2 = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=(3, 1), stride=(2, 1),
                                              padding=(1, 0), output_padding=(1, 0))

    def forward(self, x):
        _, C, _, _ = x.size()
        out1 = self.upsampling1(x[:, :C // 2])
        out2 = self.upsampling2(x[:, C // 2:])
        return torch.cat([out1, out2], dim=1)


# dilated dense block
class dilatedDenseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_layers):
        super(dilatedDenseBlock, self).__init__()

        self.input_layer = causalConv2d(in_ch, in_ch // 2, kernel_size=(3, 2), padding=(1, 1))  # channel half
        self.prelu1 = nn.PReLU()

        # dilated dense layer
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.caus_padd = ((2 ** i) // 2) * 2
            if i == 0: self.caus_padd = 1

            self.layers.append(nn.Sequential(
                causalConv2d(in_ch // 2 + i * in_ch // 2, in_ch // 2, kernel_size=(3, 2),
                             padding=(2 ** i, self.caus_padd), dilation=2 ** i, groups=in_ch // 2),
                nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=1),
                nn.GroupNorm(1, in_ch // 2, eps=1e-8),
                nn.PReLU()
            ))

        self.out_layer = causalConv2d(in_ch // 2, out_ch, kernel_size=(3, 2), padding=(1, 1))  # channel revert
        self.prelu2 = nn.PReLU()

    def forward(self, x):
        x = self.input_layer(x)  # C: in_ch//2
        x = self.prelu1(x)

        out1 = self.layers[0](x)

        out = complex_cat([out1, x], dim=1)  # C: in_ch//2 * 2
        out2 = self.layers[1](out)

        out = complex_cat([out2, out1, x], dim=1)
        out3 = self.layers[2](out)

        out = complex_cat([out3, out2, out1, x], dim=1)
        out4 = self.layers[3](out)

        out = complex_cat([out4, out3, out2, out1, x], dim=1)
        out5 = self.layers[4](out)

        out = complex_cat([out5, out4, out3, out2, out1, x], dim=1)
        out6 = self.layers[5](out)

        out = complex_cat([out6, out5, out4, out3, out2, out1, x], dim=1)
        out = self.layers[6](out)

        out = self.out_layer(out)
        out = self.prelu2(out)

        return out


# causal version of a time-frequency attention (TFA) module
# The paper of the original TFA : https://arxiv.org/abs/2111.07518
class CTFA(nn.Module):
    def __init__(self, in_ch, out_ch=16, time_seq=32):
        super(CTFA, self).__init__()
        # time attention
        self.time_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.time_conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.time_relu = nn.ReLU()
        self.time_conv2 = nn.Conv1d(out_ch, in_ch, kernel_size=1)
        self.time_sigmoid = nn.Sigmoid()

        # frequency attention
        self.freq_avg_pool = nn.AvgPool1d(time_seq, stride=1)
        self.freq_conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.freq_relu = nn.ReLU()
        self.freq_conv2 = nn.Conv2d(out_ch, in_ch, kernel_size=1)
        self.freq_sigmoid = nn.Sigmoid()

        # for real-time
        self.padd = time_seq - 1

    def forward(self, x):
        B, C, D, T = x.size()

        # time attention
        Z_T = x.permute(0, 1, 3, 2)
        Z_T = Z_T.reshape([B, C * T, D])
        FA = self.time_avg_pool(Z_T)  # [B, C*T, 1]
        FA = FA.reshape([B, C, T])
        FA = self.time_conv1(FA)
        FA = self.time_relu(FA)
        FA = self.time_conv2(FA)
        FA = self.time_sigmoid(FA)
        FA = FA.reshape([B, C, 1, T])
        FA = FA.expand(B, C, D, T)

        # frequency attention
        x_pad = functional.pad(x, [self.padd, 0, 0, 0])
        Z_F = x_pad.reshape([B, C * D, T + self.padd])
        TA = self.freq_avg_pool(Z_F)  # [B, C*F, T]
        TA = TA.reshape([B, C, D, T])
        TA = self.freq_conv1(TA)
        TA = self.freq_relu(TA)
        TA = self.freq_conv2(TA)
        TA = self.freq_sigmoid(TA)  # [B, C, D, T]

        # multiply
        TFA = TA * FA
        out = x * TFA

        return out


# Complex multi-Scale Feature Extraction (CMSFE) - e6 (for encoder part)
class CMSFEe6(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(CMSFEe6, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)
        self.en3 = CONV(mid_ch, mid_ch)
        self.en4 = CONV(mid_ch, mid_ch)
        self.en5 = CONV(mid_ch, mid_ch)
        self.en6 = CONV(mid_ch, mid_ch)

        # bottleneck
        self.ddense = dilatedDenseBlock(mid_ch, mid_ch, 7)

        # decoder (for real and imag part)
        self.de1 = CSPCONV(mid_ch, mid_ch // 2)
        self.de2 = CSPCONV(mid_ch, mid_ch // 2)
        self.de3 = CSPCONV(mid_ch, mid_ch // 2)
        self.de4 = CSPCONV(mid_ch, mid_ch // 2)
        self.de5 = CSPCONV(mid_ch, mid_ch // 2)
        self.de6 = CSPCONV(mid_ch, out_ch // 2)

        # # attention module
        self.ctfa_r = CTFA(out_ch // 2)
        self.ctfa_i = CTFA(out_ch // 2)

    def forward(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)
        out3 = self.en3(out2)
        out4 = self.en4(out3)
        out5 = self.en5(out4)
        out6 = self.en6(out5)

        # bottleneck
        out = self.ddense(out6)

        _, C, _, _ = out.size()
        real_out7 = out[:, :C // 2]
        imag_out7 = out[:, C // 2:]

        # decoder
        real_out6, imag_out6 = self.de1(torch.cat([real_out7, out6[:, :C // 2]], dim=1),
                                        torch.cat([imag_out7, out6[:, C // 2:]], dim=1))
        real_out5, imag_out5 = self.de2(torch.cat([real_out6, out5[:, :C // 2]], dim=1),
                                        torch.cat([imag_out6, out5[:, C // 2:]], dim=1))
        real_out4, imag_out4 = self.de3(torch.cat([real_out5, out4[:, :C // 2]], dim=1),
                                        torch.cat([imag_out5, out4[:, C // 2:]], dim=1))
        real_out3, imag_out3 = self.de4(torch.cat([real_out4, out3[:, :C // 2]], dim=1),
                                        torch.cat([imag_out4, out3[:, C // 2:]], dim=1))
        real_out2, imag_out2 = self.de5(torch.cat([real_out3, out2[:, :C // 2]], dim=1),
                                        torch.cat([imag_out3, out2[:, C // 2:]], dim=1))
        real_out1, imag_out1 = self.de6(torch.cat([real_out2, out1[:, :C // 2]], dim=1),
                                        torch.cat([imag_out2, out1[:, C // 2:]], dim=1))

        out = torch.cat([self.ctfa_r(real_out1), self.ctfa_i(imag_out1)], dim=1)

        out += x
        return out, real_out1, real_out2, real_out3, real_out4, real_out5, real_out6, real_out7, \
               imag_out1, imag_out2, imag_out3, imag_out4, imag_out5, imag_out6, imag_out7


# Complex multi-Scale Feature Extraction (CMSFE) - e5
class CMSFEe5(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(CMSFEe5, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)
        self.en3 = CONV(mid_ch, mid_ch)
        self.en4 = CONV(mid_ch, mid_ch)
        self.en5 = CONV(mid_ch, mid_ch)

        # bottleneck
        self.ddense = dilatedDenseBlock(mid_ch, mid_ch, 7)

        # decoder (for real and imag part)
        self.de1 = CSPCONV(mid_ch, mid_ch // 2)
        self.de2 = CSPCONV(mid_ch, mid_ch // 2)
        self.de3 = CSPCONV(mid_ch, mid_ch // 2)
        self.de4 = CSPCONV(mid_ch, mid_ch // 2)
        self.de5 = CSPCONV(mid_ch, out_ch // 2)

    def forward(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)
        out3 = self.en3(out2)
        out4 = self.en4(out3)
        out5 = self.en5(out4)

        # bottleneck
        out = self.ddense(out5)

        _, C, _, _ = out.size()
        real_out6 = out[:, :C // 2]
        imag_out6 = out[:, C // 2:]

        # decoder
        real_out5, imag_out5 = self.de1(torch.cat([real_out6, out5[:, :C // 2]], dim=1),
                                        torch.cat([imag_out6, out5[:, C // 2:]], dim=1))
        real_out4, imag_out4 = self.de2(torch.cat([real_out5, out4[:, :C // 2]], dim=1),
                                        torch.cat([imag_out5, out4[:, C // 2:]], dim=1))
        real_out3, imag_out3 = self.de3(torch.cat([real_out4, out3[:, :C // 2]], dim=1),
                                        torch.cat([imag_out4, out3[:, C // 2:]], dim=1))
        real_out2, imag_out2 = self.de4(torch.cat([real_out3, out2[:, :C // 2]], dim=1),
                                        torch.cat([imag_out3, out2[:, C // 2:]], dim=1))
        real_out1, imag_out1 = self.de5(torch.cat([real_out2, out1[:, :C // 2]], dim=1),
                                        torch.cat([imag_out2, out1[:, C // 2:]], dim=1))

        out = torch.cat([real_out1, imag_out1], dim=1)

        out += x
        return out, real_out1, real_out2, real_out3, real_out4, real_out5, real_out6, \
               imag_out1, imag_out2, imag_out3, imag_out4, imag_out5, imag_out6


# Complex multi-Scale Feature Extraction (MSFE) - e4
class CMSFEe4(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(CMSFEe4, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)
        self.en3 = CONV(mid_ch, mid_ch)
        self.en4 = CONV(mid_ch, mid_ch)

        # bottleneck
        self.ddense = dilatedDenseBlock(mid_ch, mid_ch, 7)

        # decoder (for real and imag part)
        self.de1 = CSPCONV(mid_ch, mid_ch // 2)
        self.de2 = CSPCONV(mid_ch, mid_ch // 2)
        self.de3 = CSPCONV(mid_ch, mid_ch // 2)
        self.de4 = CSPCONV(mid_ch, out_ch // 2)

    def forward(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)
        out3 = self.en3(out2)
        out4 = self.en4(out3)

        # bottleneck
        out = self.ddense(out4)

        _, C, _, _ = out.size()
        real_out5 = out[:, :C // 2]
        imag_out5 = out[:, C // 2:]

        # decoder
        real_out4, imag_out4 = self.de1(torch.cat([real_out5, out4[:, :C // 2]], dim=1),
                                        torch.cat([imag_out5, out4[:, C // 2:]], dim=1))
        real_out3, imag_out3 = self.de2(torch.cat([real_out4, out3[:, :C // 2]], dim=1),
                                        torch.cat([imag_out4, out3[:, C // 2:]], dim=1))
        real_out2, imag_out2 = self.de3(torch.cat([real_out3, out2[:, :C // 2]], dim=1),
                                        torch.cat([imag_out3, out2[:, C // 2:]], dim=1))
        real_out1, imag_out1 = self.de4(torch.cat([real_out2, out1[:, :C // 2]], dim=1),
                                        torch.cat([imag_out2, out1[:, C // 2:]], dim=1))

        out = torch.cat([real_out1, imag_out1], dim=1)

        out += x
        return out, real_out1, real_out2, real_out3, real_out4, real_out5, \
               imag_out1, imag_out2, imag_out3, imag_out4, imag_out5


# Complex multi-Scale Feature Extraction (MSFE) - e3
class CMSFEe3(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(CMSFEe3, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)
        self.en3 = CONV(mid_ch, mid_ch)

        # bottleneck
        self.ddense = dilatedDenseBlock(mid_ch, mid_ch, 7)

        # decoder (for real and imag part)
        self.de1 = CSPCONV(mid_ch, mid_ch // 2)
        self.de2 = CSPCONV(mid_ch, mid_ch // 2)
        self.de3 = CSPCONV(mid_ch, out_ch // 2)

    def forward(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)
        out3 = self.en3(out2)

        # bottleneck
        out = self.ddense(out3)

        _, C, _, _ = out.size()
        real_out4 = out[:, :C // 2]
        imag_out4 = out[:, C // 2:]

        # decoder
        real_out3, imag_out3 = self.de1(torch.cat([real_out4, out3[:, :C // 2]], dim=1),
                                        torch.cat([imag_out4, out3[:, C // 2:]], dim=1))
        real_out2, imag_out2 = self.de2(torch.cat([real_out3, out2[:, :C // 2]], dim=1),
                                        torch.cat([imag_out3, out2[:, C // 2:]], dim=1))
        real_out1, imag_out1 = self.de3(torch.cat([real_out2, out1[:, :C // 2]], dim=1),
                                        torch.cat([imag_out2, out1[:, C // 2:]], dim=1))

        out = torch.cat([real_out1, imag_out1], dim=1)

        out += x
        return out, real_out1, real_out2, real_out3, real_out4, imag_out1, imag_out2, imag_out3, imag_out4


# Multi-Scale Feature Extraction (MSFE) - d6 (for decoder part)
class MSFEd6(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(MSFEd6, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)
        self.en3 = CONV(mid_ch, mid_ch)
        self.en4 = CONV(mid_ch, mid_ch)
        self.en5 = CONV(mid_ch, mid_ch)
        self.en6 = CONV(mid_ch, mid_ch)

        # dense
        self.ddense = dilatedDenseBlock(mid_ch, mid_ch, 7)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, mid_ch)
        self.de4 = SPCONV(mid_ch * 2, mid_ch)
        self.de5 = SPCONV(mid_ch * 2, mid_ch)
        self.de6 = SPCONV(mid_ch * 2, out_ch)

        # for skip connection
        self.cat1e = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1)
        self.cat2e = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=1)
        self.cat3e = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=1)
        self.cat4e = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=1)
        self.cat5e = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=1)
        self.cat6e = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=1)

        # for skip connection
        self.cat1d = nn.Conv2d(mid_ch * 3, mid_ch * 2, kernel_size=1)
        self.cat2d = nn.Conv2d(mid_ch * 3, mid_ch * 2, kernel_size=1)
        self.cat3d = nn.Conv2d(mid_ch * 3, mid_ch * 2, kernel_size=1)
        self.cat4d = nn.Conv2d(mid_ch * 3, mid_ch * 2, kernel_size=1)
        self.cat5d = nn.Conv2d(mid_ch * 3, mid_ch * 2, kernel_size=1)
        self.cat6d = nn.Conv2d(mid_ch * 3, mid_ch * 2, kernel_size=1)

        # attention module
        self.ctfa = CTFA(out_ch)

    def forward(self, x, ed1, ed2, ed3, ed4, ed5, ed6, ed7):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(self.cat1e(torch.cat([x, ed1], dim=1)))
        out2 = self.en2(self.cat2e(torch.cat([out1, ed2], dim=1)))
        out3 = self.en3(self.cat3e(torch.cat([out2, ed3], dim=1)))
        out4 = self.en4(self.cat4e(torch.cat([out3, ed4], dim=1)))
        out5 = self.en5(self.cat5e(torch.cat([out4, ed5], dim=1)))
        out6 = self.en6(self.cat6e(torch.cat([out5, ed6], dim=1)))

        # dense
        out = self.ddense(out6)

        # decoder
        out = self.cat1d(torch.cat([out, out6, ed7], dim=1))
        out = self.de1(out)
        out = self.cat2d(torch.cat([out, out5, ed6], dim=1))
        out = self.de2(out)
        out = self.cat3d(torch.cat([out, out4, ed5], dim=1))
        out = self.de3(out)
        out = self.cat4d(torch.cat([out, out3, ed4], dim=1))
        out = self.de4(out)
        out = self.cat5d(torch.cat([out, out2, ed3], dim=1))
        out = self.de5(out)
        out = self.cat6d(torch.cat([out, out1, ed2], dim=1))
        out = self.de6(out)

        out = self.ctfa(out)

        out += x
        return out


# Multi-Scale Feature Extraction (MSFE) - d5
class MSFEd5(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(MSFEd5, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)
        self.en3 = CONV(mid_ch, mid_ch)
        self.en4 = CONV(mid_ch, mid_ch)
        self.en5 = CONV(mid_ch, mid_ch)

        # dense
        self.ddense = dilatedDenseBlock(mid_ch, mid_ch, 7)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, mid_ch)
        self.de4 = SPCONV(mid_ch * 2, mid_ch)
        self.de5 = SPCONV(mid_ch * 2, out_ch)

        # for skip connection
        self.cat1e = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1)
        self.cat2e = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=1)
        self.cat3e = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=1)
        self.cat4e = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=1)
        self.cat5e = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=1)

        # for skip connection
        self.cat1d = nn.Conv2d(mid_ch * 3, mid_ch * 2, kernel_size=1)
        self.cat2d = nn.Conv2d(mid_ch * 3, mid_ch * 2, kernel_size=1)
        self.cat3d = nn.Conv2d(mid_ch * 3, mid_ch * 2, kernel_size=1)
        self.cat4d = nn.Conv2d(mid_ch * 3, mid_ch * 2, kernel_size=1)
        self.cat5d = nn.Conv2d(mid_ch * 3, mid_ch * 2, kernel_size=1)

    def forward(self, x, ed1, ed2, ed3, ed4, ed5, ed6):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(self.cat1e(torch.cat([x, ed1], dim=1)))
        out2 = self.en2(self.cat2e(torch.cat([out1, ed2], dim=1)))
        out3 = self.en3(self.cat3e(torch.cat([out2, ed3], dim=1)))
        out4 = self.en4(self.cat4e(torch.cat([out3, ed4], dim=1)))
        out5 = self.en5(self.cat5e(torch.cat([out4, ed5], dim=1)))

        # dense
        out = self.ddense(out5)

        # decoder
        out = self.cat1d(torch.cat([out, out5, ed6], dim=1))
        out = self.de1(out)
        out = self.cat2d(torch.cat([out, out4, ed5], dim=1))
        out = self.de2(out)
        out = self.cat3d(torch.cat([out, out3, ed4], dim=1))
        out = self.de3(out)
        out = self.cat4d(torch.cat([out, out2, ed3], dim=1))
        out = self.de4(out)
        out = self.cat5d(torch.cat([out, out1, ed2], dim=1))
        out = self.de5(out)

        out += x
        return out


# Multi-Scale Feature Extraction (MSFE) - d4
class MSFEd4(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(MSFEd4, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)
        self.en3 = CONV(mid_ch, mid_ch)
        self.en4 = CONV(mid_ch, mid_ch)

        # dense
        self.ddense = dilatedDenseBlock(mid_ch, mid_ch, 7)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, mid_ch)
        self.de4 = SPCONV(mid_ch * 2, out_ch)

        # for skip connection
        self.cat1e = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1)
        self.cat2e = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=1)
        self.cat3e = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=1)
        self.cat4e = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=1)

        # for skip connection
        self.cat1d = nn.Conv2d(mid_ch * 3, mid_ch * 2, kernel_size=1)
        self.cat2d = nn.Conv2d(mid_ch * 3, mid_ch * 2, kernel_size=1)
        self.cat3d = nn.Conv2d(mid_ch * 3, mid_ch * 2, kernel_size=1)
        self.cat4d = nn.Conv2d(mid_ch * 3, mid_ch * 2, kernel_size=1)

    def forward(self, x, ed1, ed2, ed3, ed4, ed5):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(self.cat1e(torch.cat([x, ed1], dim=1)))
        out2 = self.en2(self.cat2e(torch.cat([out1, ed2], dim=1)))
        out3 = self.en3(self.cat3e(torch.cat([out2, ed3], dim=1)))
        out4 = self.en4(self.cat4e(torch.cat([out3, ed4], dim=1)))

        out = self.ddense(out4)

        # decoder
        out = self.cat1d(torch.cat([out, out4, ed5], dim=1))
        out = self.de1(out)
        out = self.cat2d(torch.cat([out, out3, ed4], dim=1))
        out = self.de2(out)
        out = self.cat3d(torch.cat([out, out2, ed3], dim=1))
        out = self.de3(out)
        out = self.cat4d(torch.cat([out, out1, ed2], dim=1))
        out = self.de4(out)

        out += x
        return out


# Multi-Scale Feature Extraction (MSFE) - d3
class MSFEd3(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(MSFEd3, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)
        self.en3 = CONV(mid_ch, mid_ch)

        # dense
        self.ddense = dilatedDenseBlock(mid_ch, mid_ch, 7)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, out_ch)

        # for skip connection
        self.cat1e = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1)
        self.cat2e = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=1)
        self.cat3e = nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=1)

        # for skip connection
        self.cat1d = nn.Conv2d(mid_ch * 3, mid_ch * 2, kernel_size=1)
        self.cat2d = nn.Conv2d(mid_ch * 3, mid_ch * 2, kernel_size=1)
        self.cat3d = nn.Conv2d(mid_ch * 3, mid_ch * 2, kernel_size=1)

    def forward(self, x, ed1, ed2, ed3, ed4):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(self.cat1e(torch.cat([x, ed1], dim=1)))
        out2 = self.en2(self.cat2e(torch.cat([out1, ed2], dim=1)))
        out3 = self.en3(self.cat3e(torch.cat([out2, ed3], dim=1)))

        # dense
        out = self.ddense(out3)

        # decoder
        out = self.cat1d(torch.cat([out, out3, ed4], dim=1))
        out = self.de1(out)
        out = self.cat2d(torch.cat([out, out2, ed3], dim=1))
        out = self.de2(out)
        out = self.cat3d(torch.cat([out, out1, ed2], dim=1))
        out = self.de3(out)

        out += x
        return out


class CNUNet(nn.Module):

    def __init__(self, in_ch=2, mid_ch=32, out_ch=64,
                 WIN_LEN=400, HOP_LEN=100, FFT_LEN=512):
        super(CNUNet, self).__init__()

        # input layer
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CMSFEe6(out_ch, mid_ch, out_ch)
        self.down_sampling1 = down_sampling_half(out_ch)

        self.en2 = CMSFEe5(out_ch, mid_ch, out_ch)
        self.down_sampling2 = down_sampling_half(out_ch)

        self.en3 = CMSFEe4(out_ch, mid_ch, out_ch)
        self.down_sampling3 = down_sampling_half(out_ch)

        self.en4 = CMSFEe4(out_ch, mid_ch, out_ch)
        self.down_sampling4 = down_sampling_half(out_ch)

        self.en5 = CMSFEe4(out_ch, mid_ch, out_ch)
        self.down_sampling5 = down_sampling_half(out_ch)

        self.en6 = CMSFEe3(out_ch, mid_ch, out_ch)
        self.down_sampling6 = down_sampling_half(out_ch)

        # Bottleneck block
        self.DDense = nn.Sequential(
            dilatedDenseBlock(out_ch, out_ch, 7)
        )

        # decoder (real)
        self.upsampling_1r = upsampling_half(out_ch // 2)
        self.de1r = MSFEd3(out_ch, mid_ch // 2, out_ch // 2)

        self.upsampling_2r = upsampling_half(out_ch // 2)
        self.de2r = MSFEd4(out_ch, mid_ch // 2, out_ch // 2)

        self.upsampling_3r = upsampling_half(out_ch // 2)
        self.de3r = MSFEd4(out_ch, mid_ch // 2, out_ch // 2)

        self.upsampling_4r = upsampling_half(out_ch // 2)
        self.de4r = MSFEd4(out_ch, mid_ch // 2, out_ch // 2)

        self.upsampling_5r = upsampling_half(out_ch // 2)
        self.de5r = MSFEd5(out_ch, mid_ch // 2, out_ch // 2)

        self.upsampling_6r = upsampling_half(out_ch // 2)
        self.de6r = MSFEd6(out_ch, mid_ch // 2, out_ch // 2)

        # decoder (imag)
        self.upsampling_1i = upsampling_half(out_ch // 2)
        self.de1i = MSFEd3(out_ch, mid_ch // 2, out_ch // 2)

        self.upsampling_2i = upsampling_half(out_ch // 2)
        self.de2i = MSFEd4(out_ch, mid_ch // 2, out_ch // 2)

        self.upsampling_3i = upsampling_half(out_ch // 2)
        self.de3i = MSFEd4(out_ch, mid_ch // 2, out_ch // 2)

        self.upsampling_4i = upsampling_half(out_ch // 2)
        self.de4i = MSFEd4(out_ch, mid_ch // 2, out_ch // 2)

        self.upsampling_5i = upsampling_half(out_ch // 2)
        self.de5i = MSFEd5(out_ch, mid_ch // 2, out_ch // 2)

        self.upsampling_6i = upsampling_half(out_ch // 2)
        self.de6i = MSFEd6(out_ch, mid_ch // 2, out_ch // 2)

        # output layer
        self.output_layer = nn.Conv2d(out_ch, in_ch, kernel_size=1)

        # for feature extract
        self.cstft = ConvSTFT(WIN_LEN, HOP_LEN, FFT_LEN, feature_type='complex')
        self.cistft = ConviSTFT(WIN_LEN, HOP_LEN, FFT_LEN, feature_type='complex')

    def forward(self, x):
        # STFT
        specs = self.cstft(x)  # [B, F, T]
        real = specs[:, :257, :]
        imag = specs[:, 257:, :]
        hx = torch.stack([real, imag], dim=1)
        hx = hx[:, :, 1:]

        # input layer
        hx = self.input_layer(hx)

        # encoder stage 1
        hx1, hx1_1r, hx1_2r, hx1_3r, hx1_4r, hx1_5r, hx1_6r, hx1_7r, \
        hx1_1i, hx1_2i, hx1_3i, hx1_4i, hx1_5i, hx1_6i, hx1_7i = self.en1(hx)
        hx = self.down_sampling1(hx1)

        # encoder stage 2
        hx2, hx2_1r, hx2_2r, hx2_3r, hx2_4r, hx2_5r, hx2_6r, \
        hx2_1i, hx2_2i, hx2_3i, hx2_4i, hx2_5i, hx2_6i = self.en2(hx)
        hx = self.down_sampling2(hx2)

        # encoder stage 3
        hx3, hx3_1r, hx3_2r, hx3_3r, hx3_4r, hx3_5r, \
        hx3_1i, hx3_2i, hx3_3i, hx3_4i, hx3_5i = self.en3(hx)
        hx = self.down_sampling3(hx3)

        # encoder stage 4
        hx4, hx4_1r, hx4_2r, hx4_3r, hx4_4r, hx4_5r, \
        hx4_1i, hx4_2i, hx4_3i, hx4_4i, hx4_5i = self.en4(hx)
        hx = self.down_sampling4(hx4)

        # encoder stage 5
        hx5, hx5_1r, hx5_2r, hx5_3r, hx5_4r, hx5_5r, \
        hx5_1i, hx5_2i, hx5_3i, hx5_4i, hx5_5i = self.en5(hx)
        hx = self.down_sampling5(hx5)

        # encoder stage 6
        hx6, hx6_1r, hx6_2r, hx6_3r, hx6_4r, \
        hx6_1i, hx6_2i, hx6_3i, hx6_4i = self.en6(hx)
        hx = self.down_sampling6(hx6)

        # dilated dense block
        out = self.DDense(hx)

        # decoder stage 1
        real_out = self.upsampling_1r(out[:, :out.size(1) // 2])
        imag_out = self.upsampling_1i(out[:, out.size(1) // 2:])
        real_out = self.de1r(torch.cat([real_out, hx6[:, :hx6.size(1) // 2]], dim=1), hx6_1r, hx6_2r, hx6_3r, hx6_4r)
        imag_out = self.de1i(torch.cat([imag_out, hx6[:, hx6.size(1) // 2:]], dim=1), hx6_1i, hx6_2i, hx6_3i, hx6_4i)

        # decoder stage 2
        real_out = self.upsampling_2r(real_out)
        imag_out = self.upsampling_2i(imag_out)
        real_out = self.de2r(torch.cat([real_out, hx5[:, :hx5.size(1) // 2]], dim=1), hx5_1r, hx5_2r, hx5_3r, hx5_4r,
                             hx5_5r)
        imag_out = self.de2i(torch.cat([imag_out, hx5[:, hx5.size(1) // 2:]], dim=1), hx5_1i, hx5_2i, hx5_3i, hx5_4i,
                             hx5_5i)

        # decoder stage 3
        real_out = self.upsampling_3r(real_out)
        imag_out = self.upsampling_3i(imag_out)
        real_out = self.de3r(torch.cat([real_out, hx4[:, :hx4.size(1) // 2]], dim=1), hx4_1r, hx4_2r, hx4_3r, hx4_4r,
                             hx4_5r)
        imag_out = self.de3i(torch.cat([imag_out, hx4[:, hx4.size(1) // 2:]], dim=1), hx4_1i, hx4_2i, hx4_3i, hx4_4r,
                             hx4_5i)

        # decoder stage 4
        real_out = self.upsampling_4r(real_out)
        imag_out = self.upsampling_4i(imag_out)
        real_out = self.de4r(torch.cat([real_out, hx3[:, :hx3.size(1) // 2]], dim=1), hx3_1r, hx3_2r, hx3_3r, hx3_4r,
                             hx3_5r)
        imag_out = self.de4i(torch.cat([imag_out, hx3[:, hx3.size(1) // 2:]], dim=1), hx3_1i, hx3_2i, hx3_3i, hx3_4r,
                             hx3_5i)

        # decoder stage 5
        real_out = self.upsampling_5r(real_out)
        imag_out = self.upsampling_5i(imag_out)
        real_out = self.de5r(torch.cat([real_out, hx2[:, :hx2.size(1) // 2]], dim=1), hx2_1r, hx2_2r, hx2_3r, hx2_4i,
                             hx2_5r, hx2_6r)
        imag_out = self.de5i(torch.cat([imag_out, hx2[:, hx2.size(1) // 2:]], dim=1), hx2_1i, hx2_2i, hx2_3i, hx2_4i,
                             hx2_5i, hx2_6i)

        # decoder stage 6
        real_out = self.upsampling_6r(real_out)
        imag_out = self.upsampling_6i(imag_out)
        real_out = self.de6r(torch.cat([real_out, hx1[:, :hx1.size(1) // 2]], dim=1), hx1_1r, hx1_2r, hx1_3r, hx1_4r,
                             hx1_5r, hx1_6r, hx1_7r)
        imag_out = self.de6i(torch.cat([imag_out, hx1[:, hx1.size(1) // 2:]], dim=1), hx1_1i, hx1_2i, hx1_3i, hx1_4i,
                             hx1_5i, hx1_6i, hx1_7i)

        # output layer
        out = self.output_layer(torch.cat([real_out, imag_out], dim=1))

        out = functional.pad(out, [0, 0, 1, 0])
        real_out = out[:, 0]
        imag_out = out[:, 1]
        out = torch.cat([real_out, imag_out], dim=1)

        # ISTFT
        out_wav = self.cistft(out).squeeze(1)
        out_wav = torch.clamp_(out_wav, -1, 1)  # clipping [-1, 1]
        return out_wav
