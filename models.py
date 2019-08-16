"""
Model definitions for the Crosswise Sparse Convolutional Autoencoder.
See https://www3.cs.stonybrook.edu/~cvl/content/papers/2018/Le_PR19.pdf
"""

import  torch
nn      = torch.nn
F       = nn.functional
from    torch               import  optim
import  torchvision.utils   as      vutils

import  matplotlib.pyplot   as      plt
import  os
import  numpy               as      np
import  sys
import  imageio


class FirstConvBlock(nn.Module):
    def __init__(self, options):
        super(FirstConvBlock, self).__init__()

        self.ndf        = options.ndf
        self.nc         = options.nc

        # === Convolutional block ===
        self.conv1      = nn.Conv2d(self.nc, self.ndf, 7, stride=1, padding=3)
        self.bn1        = nn.BatchNorm2d(self.ndf)

        self.conv2      = nn.Conv2d(self.ndf, self.ndf, 3, stride=1, padding=1)
        self.bn2        = nn.BatchNorm2d(self.ndf)

        self.conv3      = nn.Conv2d(self.ndf, self.ndf * 2, 3, stride=2, padding=1)
        self.bn3        = nn.BatchNorm2d(self.ndf * 2)

        self.conv4      = nn.Conv2d(self.ndf * 2, self.ndf * 2, 3, stride=1, padding=1)
        self.bn4        = nn.BatchNorm2d(self.ndf * 2)

        self.conv5      = nn.Conv2d(self.ndf * 2, self.ndf * 4, 3, stride=2, padding=1)
        self.bn5        = nn.BatchNorm2d(self.ndf * 4)

        self.conv6      = nn.Conv2d(self.ndf * 4, self.ndf * 4, 3, stride=1, padding=1)
        self.bn6        = nn.BatchNorm2d(self.ndf * 4)
        # ===========================

        self.relu       = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, x):
        # Pass x through the entire convolutional block. 
        y               = self.relu(self.bn1(self.conv1(x)))
        y               = self.relu(self.bn2(self.conv2(y)))
        y               = self.relu(self.bn3(self.conv3(y)))
        y               = self.relu(self.bn4(self.conv4(y)))
        y               = self.relu(self.bn5(self.conv5(y)))
        y               = self.relu(self.bn6(self.conv6(y)))

        return y


class DetectionBranch(nn.Module):
    def __init__(self, options):
        super(DetectionBranch, self).__init__()

        self.beta       = options.beta
        self.r          = options.r
        self.p          = options.p
        self.nc         = options.ndf * 4
        # Registering tau and alpha as buffers records them as 
        #   important members and puts them as untrainable
        #   parameters in the state dict, so that when state dicts
        #   are saved and loaded, these parameters are also preserved.
        self.register_buffer('tau', torch.Tensor([0.0]).float().squeeze())
        self.register_buffer('alpha', torch.FloatTensor([1.0]).float().squeeze())

        # === Convolutional block ===
        self.conv1      = nn.Conv2d(self.nc, self.nc // 4, 1, stride=1, padding=0)
        self.bn1        = nn.BatchNorm2d(self.nc // 4)

        self.conv2      = nn.Conv2d(self.nc // 4, 1, 1, stride=1, padding=0)
        # ===========================

        self.relu       = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, x, train=True):
        # Pass through convolutional layers to get a dense detection map. 
        D_prime         = self.relu(self.bn1(self.conv1(x)))
        D_prime         = self.conv2(D_prime)

        if train:
            # Use tau derived from the dense detection map only in the training phase. 
            #   In the val/test phase, we instead use the tau learnt so far.

            t_idx       = int(np.floor(self.p / 100.0 * (x.size(1) * x.size(2))))
            # Train with tau if in training phase. Else, use the learnt tau. 
            _v          = D_prime.view(x.size(0), -1).detach()         # .detach() is necessary. 
            _v, _       = torch.sort(_v, dim=1, descending=True)
            tau         = torch.mean(_v[:, t_idx])
    
            # Update tau. The following rule appears in the paper. 
            self.tau    = 0.9 * self.tau + 0.1 * tau

            # Another rule worth trying---keep increasing the weight of the tau recorded so far,
            #   by decaying self.alpha
#            self.tau    = (1 - self.alpha) * self.tau + (self.alpha) * tau
            # Decay alpha---only used with the second update rule. 
            self.alpha  = (1 + self.alpha) ** self.beta - 1

        # Apply sigmoid to get a sparse detection map. 
        D               = torch.sigmoid(self.r * (D_prime - self.tau))
        return D
        

class ForegroundBranch(nn.Module):
    def __init__(self, options):
        super(ForegroundBranch, self).__init__()

        self.nc         = options.ndf * 4

        # === Convolutional block ===
        self.conv1      = nn.Conv2d(self.nc, self.nc // 2, 1, stride=1, padding=0)
        self.bn1        = nn.BatchNorm2d(self.nc // 2)

        self.conv2      = nn.Conv2d(self.nc // 2, self.nc // 4, 1, stride=1, padding=0)
        self.bn2        = nn.BatchNorm2d(self.nc // 4)
        # ===========================

        self.relu       = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, M, D):
        # D comes from DetectionBranch. It is supposed to be the sparse detection map. 
        X_prime         = self.relu(self.bn1(self.conv1(M)))
        X_prime         = self.relu(self.bn2(self.conv2(X_prime)))

        X               = X_prime * D
        return X

class BackgroundBranch(nn.Module):
    def __init__(self, options):
        super(BackgroundBranch, self).__init__()

        self.nc         = options.ndf * 4

        # === Convolutional block ===
        self.conv1      = nn.Conv2d(self.nc, self.nc // 4, 5, stride=2, padding=2)
        self.bn1        = nn.BatchNorm2d(self.nc // 4)

        self.conv2      = nn.Conv2d(self.nc // 4, self.nc // 8, 3, stride=2, padding=1)
        self.bn2        = nn.BatchNorm2d(self.nc // 8)

        self.conv3      = nn.Conv2d(self.nc // 8, 5, 3, stride=2, padding=1)
        self.bn3        = nn.BatchNorm2d(5)
        # ===========================

        self.relu       = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, M):
        y               = self.relu(self.bn1(self.conv1(M)))
        y               = self.relu(self.bn2(self.conv2(y)))
        y               = self.relu(self.bn3(self.conv3(y)))
        return y


class ForegroundDeconvBranch(nn.Module):
    def __init__(self, options):
        super(ForegroundDeconvBranch, self).__init__()

        self.ndf        = options.ndf
        self.nc         = options.ndf

        self.upsample   = nn.Upsample(scale_factor=2, mode='nearest')

        # === Convolutional block ===
        # This ``convolutional'' block uses upsampling instead of deconvolutions. 
        # See https://distill.pub/2016/deconv-checkerboard/
        self.conv1      = nn.Conv2d(self.nc, self.ndf * 2, 3, stride=1, padding=1)
        self.bn1        = nn.BatchNorm2d(self.ndf * 2)

        self.conv2      = nn.Conv2d(self.ndf * 2, self.ndf * 2, 3, stride=1, padding=1)
        self.bn2        = nn.BatchNorm2d(self.ndf * 2)
       
        self.conv3      = nn.Conv2d(self.ndf * 2, self.ndf, 3, stride=1, padding=1)
        self.bn3        = nn.BatchNorm2d(self.ndf)

        self.conv4      = nn.Conv2d(self.ndf, self.ndf, 3, stride=1, padding=1)
        self.bn4        = nn.BatchNorm2d(self.ndf)

        self.conv5      = nn.Conv2d(self.ndf, options.nc, 3, stride=1, padding=1)
        # ===========================
        
        self.relu       = F.relu

    def forward(self, M):
        R               = self.upsample(M)
        R               = self.relu(self.bn1(self.conv1(R)))
        R               = self.relu(self.bn2(self.conv2(R)))

        R               = self.upsample(R)
        R               = self.relu(self.bn3(self.conv3(R)))
        R               = self.relu(self.bn4(self.conv4(R)))

        R               = self.conv5(R)

        return R


class BackgroundDeconvBranch(nn.Module):
    def __init__(self, options):
        super(BackgroundDeconvBranch, self).__init__()

        self.ndf        = options.ndf
        self.nc         = 5

        self.upsample   = nn.Upsample(scale_factor=2, mode='nearest')

        # === Convolutional block ===
        # This ``convolutional'' block uses upsampling instead of deconvolutions. 
        # See https://distill.pub/2016/deconv-checkerboard/
        self.conv1      = nn.Conv2d(self.nc, self.ndf * 16, 3, stride=1, padding=1)
        self.bn1        = nn.BatchNorm2d(self.ndf * 16)

        self.conv2      = nn.Conv2d(self.ndf * 16, self.ndf * 8, 3, stride=1, padding=1)
        self.bn2        = nn.BatchNorm2d(self.ndf * 8)

        self.conv3      = nn.Conv2d(self.ndf * 8, self.ndf * 4, 3, stride=1, padding=1)
        self.bn3        = nn.BatchNorm2d(self.ndf * 4)

        self.conv4      = nn.Conv2d(self.ndf * 4, self.ndf * 2, 3, stride=1, padding=1)
        self.bn4        = nn.BatchNorm2d(self.ndf * 2)

        self.conv5      = nn.Conv2d(self.ndf * 2, self.ndf, 3, stride=1, padding=1)
        self.bn5        = nn.BatchNorm2d(self.ndf)

        self.conv6      = nn.Conv2d(self.ndf, options.nc, 3, stride=1, padding=1)
        # ===========================

        self.relu       = F.relu

    def forward(self, M):
        B               = self.upsample(M)
        B               = self.relu(self.bn1(self.conv1(B)))

        B               = self.upsample(B)
        B               = self.relu(self.bn2(self.conv2(B)))

        B               = self.upsample(B)
        B               = self.relu(self.bn3(self.conv3(B)))

        B               = self.upsample(B)
        B               = self.relu(self.bn4(self.conv4(B)))

        B               = self.upsample(B)
        B               = self.relu(self.bn5(self.conv5(B)))

        B               = self.conv6(B)
        return B


class CrosswiseSparseCAE(nn.Module):
    """
    Full Crosswise Sparse Convolutional Auto-Encoder. 
    Puts together all modules to form an end-to-end model. 
    """
    def __init__(self, options):
        super(CrosswiseSparseCAE, self).__init__()

        self.first_conv_block   = FirstConvBlock(options)
        self.detection_branch   = DetectionBranch(options)
        self.foreground_branch  = ForegroundBranch(options)
        self.background_branch  = BackgroundBranch(options)

        self.foreground_deconv  = ForegroundDeconvBranch(options)
        self.background_deconv  = BackgroundDeconvBranch(options)

    def forward(self, img, train=True):
        return_dict             = {}

        c_feats                 = self.first_conv_block(img)

        # The detection branch takes an extra argument called train.
        # In the val/test phase, the value of tau is not updated, but instead
        #   the value computed so far from training examples is used. 
        detection_map           = self.detection_branch(c_feats, train=train)
        fg_feats                = self.foreground_branch(c_feats, detection_map)
        
        bg_feats                = self.background_branch(c_feats)

        foreground              = self.foreground_deconv(fg_feats)
        background              = self.background_deconv(bg_feats)

        return_dict['c_feats']  = c_feats
        return_dict['detection_map'] = detection_map
        return_dict['fg_feats'] = fg_feats
        return_dict['bg_feats'] = bg_feats
        return_dict['foreground'] = foreground
        return_dict['background'] = background
        return_dict['recon']    = foreground + background

        return return_dict

