import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
from modules import *
import numpy as np



class ExpandingLayer(nn.Module):
    def __init__(self,n_classes, img_depth, h_w):
        super(ExpandingLayer, self).__init__()
        self.hw = h_w
        self.img_depth = img_depth
        self.n_classes = n_classes
        self.expand = nn.Linear(n_classes, self.img_depth * h_w * h_w)
        
    def forward(self, x):
        x = self.expand(x)
        x = x.view(-1, self.img_depth, self.hw, self.hw)
        return x


class DiffusionModel(pl.LightningModule):
    def __init__(self, in_size, t_range, img_depth, cdf=False):
        super().__init__()
        self.beta_small = 1e-4
        self.beta_large = 0.02
        self.t_range = t_range
        self.in_size = in_size
        self.cfg_drop_rate = 0.5
        self.n_classes = 10
        self.img_depth = img_depth
        self.h_w = 32
        self.cdf = cdf
        self.cdf_scale = 1

        
        bilinear = True
        self.inc = DoubleConv(img_depth, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, img_depth)
        self.sa1 = SAWrapper(256, 8)
        self.sa2 = SAWrapper(256, 4)
        self.sa3 = SAWrapper(128, 8)

        if self.cdf:
            self.label_layer_1 = ExpandingLayer(self.n_classes, 64, self.h_w)
            self.label_layer_2 = ExpandingLayer(self.n_classes, 128, self.h_w // 2)
            self.label_layer_3 = ExpandingLayer(self.n_classes, 256, self.h_w // 4)
            self.label_layer_4 = ExpandingLayer(self.n_classes, 256, self.h_w // (4 * factor))

    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1, 1).repeat(1, 1, embed_size, embed_size)

    def forward(self, x, t, c=None, cfg_mask=None):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """
        BS, C, H, W = x.shape
        device = x.device
        if self.cdf and c is not None:
            c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
            c_label = c * cfg_mask
            
        x1 = self.inc(x)
        if self.cdf and c is not None:
            x1 = x1 + self.label_layer_1(c_label)
        x2 = self.down1(x1) + self.pos_encoding(t, 128, 16)
        if self.cdf and c is not None:
            x2 = x2 + self.label_layer_2(c_label)
        x3 = self.down2(x2) + self.pos_encoding(t, 256, 8)
        if self.cdf and c is not None:
            x3 = x3 + self.label_layer_3(c_label)
        x3 = self.sa1(x3)
        x4 = self.down3(x3) + self.pos_encoding(t, 256, 4)
        if self.cdf and c is not None:
            x4 = x4 + self.label_layer_4(c_label)
        x4 = self.sa2(x4)
        x = self.up1(x4, x3) + self.pos_encoding(t, 128, 8)
        x = self.sa3(x)
        x = self.up2(x, x2) + self.pos_encoding(t, 64, 16)
        x = self.up3(x, x1) + self.pos_encoding(t, 64, 32)

        output = self.outc(x)
    
        return output

    def beta(self, t):
        return self.beta_small + (t / self.t_range) * (
            self.beta_large - self.beta_small
        )

    def alpha(self, t):
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        return math.prod([self.alpha(j) for j in range(t)])

    def get_loss(self, batch, batch_idx):
        """
        Corresponds to Algorithm 1 from (Ho et al., 2020).
        """
        data, labels = batch
        device = data.device
        BS = data.shape[0]
        ts = torch.randint(0, self.t_range, [data.shape[0]], device=self.device)
        noise_imgs = []
        epsilons = torch.randn(data.shape, device=self.device)
        for i in range(len(ts)):
            a_hat = self.alpha_bar(ts[i])
            noise_imgs.append(
                (math.sqrt(a_hat) * data[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
            )
        noise_imgs = torch.stack(noise_imgs, dim=0)
        cfg_mask = None
        if self.cdf:
            cfg_mask = torch.bernoulli((1 - self.cfg_drop_rate) * torch.ones(BS, 1, device=device)).repeat(1, self.n_classes)
        else:
            labels = None
        
        e_hat = self.forward(noise_imgs, ts.unsqueeze(-1).type(torch.float), c=labels, cfg_mask=cfg_mask)
        loss = nn.functional.mse_loss(
            e_hat.reshape(-1, self.in_size), epsilons.reshape(-1, self.in_size)
        )
        return loss

    def denoise_sample(self, x, t, c=None, labels=None):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        if not self.cdf:
            with torch.no_grad():
                BS = x.shape[0]
                device = x.device
                if t > 1:
                    z = torch.randn(x.shape)
                else:
                    z = 0
                e_hat = self.forward(x, t.view(1, 1).repeat(x.shape[0], 1))
                pre_scale = 1 / math.sqrt(self.alpha(t))
                e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
                post_sigma = math.sqrt(self.beta(t)) * z
                x = pre_scale * (x - e_scale * e_hat) + post_sigma
                return x
        else:        
            with torch.no_grad():
                assert c is not None, "c is None"
                BS = x.shape[0]
                device = x.device
                if t > 1:
                    z = torch.randn(x.shape)
                else:
                    z = 0
                cfg_mask = torch.bernoulli((1 - self.cfg_drop_rate) * torch.ones(BS, 1, device=device)).repeat(1, self.n_classes)
                predicted_noise = self.forward(x, t.view(1, 1).repeat(x.shape[0], 1),c=c, cfg_mask=cfg_mask)
                
                uncondition_predicted_noise = self.forward(x, t.view(1, 1).repeat(x.shape[0], 1), c=None, cfg_mask=None)

                e_hat = (1 + self.cdf_scale) * predicted_noise - self.cdf_scale * uncondition_predicted_noise
                
                pre_scale = 1 / math.sqrt(self.alpha(t))
                e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
                post_sigma = math.sqrt(self.beta(t)) * z
                x = pre_scale * (x - e_scale * e_hat) + post_sigma
                return x

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("val/loss", loss)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer
