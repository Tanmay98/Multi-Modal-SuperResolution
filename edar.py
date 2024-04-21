import common

import torch
import torch.nn as nn

class EDAR(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(EDAR, self).__init__()

        n_resblock = 8
        n_feats = 64
        kernel_size = 3

        #DIV 2K mean
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(rgb_mean, rgb_std)

        # define head module
        m_head = [conv(3, n_feats, kernel_size)]

        # define body module
        m_res = []
        m_att = []
        m_dynamic_conv = []
        m_resnext = []
        m_res.append(common.ResBlock(conv, n_feats, kernel_size))
        m_att.append(common.conv_1x1(in_channels=64, out_channels=512))
        m_dynamic_conv.append(common.DynamicConvolutionModule())

        m_resnext = [
            conv(512, 64, kernel_size)
        ]
        # define tail module
        m_tail = [
            conv(512, 3, kernel_size)
        ]

        self.add_mean = common.MeanShift(rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.res = nn.Sequential(*m_res)
        self.att = nn.Sequential(*m_att)
        self.dynamic_conv = nn.Sequential(*m_dynamic_conv)
        self.resnext = nn.Sequential(*m_resnext)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, dummy_text_emb):
        x = self.sub_mean(x)
        x = self.head(x)

        # 1 Residual Block
        res = self.res(x)
        res += x

        # Creating Multimodal mask
        f_lr = self.att(res)
        f_lr_flat = f_lr.view(f_lr.size(0), f_lr.size(1), -1)
        f_lr_flat = torch.permute(f_lr_flat, (0,2,1))

        dummy_text_emb_transpose = torch.permute(dummy_text_emb, (0,2,1))
        multimodal_mask_matrix = torch.bmm(f_lr_flat, dummy_text_emb_transpose)
        f_lr_enh = multimodal_mask_matrix * f_lr_flat
        
        # Dynamic convolution with text guidance
        f_o = self.dynamic_conv((dummy_text_emb, f_lr_enh))

        f_o_ = torch.permute(f_o, (0,2,1))
        f_o_reshaped = f_o.reshape(f_o_.size(0), 512, 48, 48)

        x = self.resnext(f_o_reshaped)

        # 2 Residual Block
        res = self.res(x)
        res += x

        # Creating Multimodal mask
        f_lr = self.att(res)
        f_lr_flat = f_lr.view(f_lr.size(0), f_lr.size(1), -1)
        f_lr_flat = torch.permute(f_lr_flat, (0,2,1))

        dummy_text_emb_transpose = torch.permute(dummy_text_emb, (0,2,1))
        multimodal_mask_matrix = torch.bmm(f_lr_flat, dummy_text_emb_transpose)
        f_lr_enh = multimodal_mask_matrix * f_lr_flat

        # Dynamic convolution with text guidance
        f_o = self.dynamic_conv((dummy_text_emb, f_lr_enh))

        f_o_ = torch.permute(f_o, (0,2,1))
        f_o_reshaped = f_o.reshape(f_o_.size(0), 512, 48, 48)

        x = self.tail(f_o_reshaped)
        x = self.add_mean(x)

        return torch.clamp(x,0.0,1.0)