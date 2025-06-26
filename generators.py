import torch
import torch.nn as nn
import torch.nn.functional as F
from quaternion_layers import QuaternionConv,QuaternionTransposeConv

class NLBlockND_cross(nn.Module):
    # Our implementation of the attention block referenced https://github.com/tea1528/Non-Local-NN-Pytorch

    def __init__(self, in_channels=32, inter_channels=None, mode='embedded',
                 dimension=1, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        """实现非局部块，使用4种不同的对数函数，但不包括下采样技巧。
        参数：
        in_channels：原始通道大小（论文中为1024）
        inter_channels：未指定时，块内部的通道大小将被缩减为一半（论文中为512）
        mode：支持高斯、嵌入高斯、点积和拼接
        dimension：可以是1（时间）、2（空间）或3（时空）
        bn_layer：是否添加批量归一化层
        """
        super(NLBlockND_cross, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block通道大小减少到块内部的一半
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions为不同的维度分配适当的卷积，最大池和批处理规范层
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1本文中的函数g，该函数经过卷积，内核大小为1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping从论文4.1节开始，对BN进行参数初始化，确保非局部块的初始状态为恒等映射
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture从论文的3.3节开始，通过将Wz初始化为0，该块可以插入到任何现有的体系结构中w
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian定义除了高斯函数之外所有运算的和
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x_thisBranch, x_otherBranch):
        #x_thisBranch for g and theta
        #x_otherBranch for phi
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """
        # print(x_thisBranch.shape)


        batch_size = x_thisBranch.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation这种重塑和排列来自原始Caffe2实现中的spacetime_nonlocal函数
        g_x = self.g(x_thisBranch).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x_thisBranch.view(batch_size, self.in_channels, -1)
            phi_x = x_otherBranch.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, -1)
            # theta_x = theta_x.permute(0, 2, 1)
            phi_x = phi_x.permute(0, 2, 1)
            f = torch.matmul(phi_x, theta_x)

        # elif self.mode == "concatenate":
        else: #default as concatenate
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory这里的contious只分配连续的内存块
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x_thisBranch.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x_thisBranch
        z=z.view(1,512)
        return z

class ACBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ACBlock, self).__init__()
        self.conv1x3 = nn.Conv2d(in_channels, out_channels, (1, 3), 1, (0, 1))
        self.conv3x1 = nn.Conv2d(in_channels, out_channels, (3, 1), 1, (1, 0))
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, (3, 3), 1, (1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(1)
        x = x.to(torch.float32)
        conv3x1 = self.conv3x1(x)
        conv1x3 = self.conv1x3(x)
        conv3x3 = self.conv3x3(x)
        output = (conv3x1 + conv1x3 + conv3x3).view(3, -1)
        output = self.relu(output)
        return output

class QCFE(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, relu=False, transpose=False):
        super(QCFE, self).__init__()
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(QuaternionTransposeConv(in_channel, out_channel, kernel_size, padding=padding, stride=stride,
                                                  bias=bias))

        else:
            layers.append(
                QuaternionConv(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))

        if relu:
            layers.append(nn.GELU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class conv_att(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(conv_att, self).__init__()
        self.att1 = NLBlockND_cross(in_channels=32, dimension=1)
        self.att2 = NLBlockND_cross(in_channels=32, dimension=1)
        self.att3 = NLBlockND_cross(in_channels=32, dimension=1)
        self.conv = ACBlock(1,1)
        self.quaternion_conv = QCFE(in_channel=4,out_channel=4,kernel_size=3,stride=1)
        self.linear_layer = nn.Linear(in_features=6, out_features=3)

    def forward(self,x):
        x_img = x[0].view(1, 32,16)
        x_rna = x[1].view(1, 32,16)
        x_cli = x[2].view(1, 32,16)
        fuse1=self.att1(x_img,x_cli)
        fuse2=self.att2(x_img,x_rna)
        fuse3=self.att3(x_cli,x_rna)
        # f1=fuse1+fuse2
        # f2=fuse2+fuse3
        # f3=fuse1+fuse3
        # fuse = torch.cat((f1,f2,f3),0)
        fuse = torch.cat((fuse1,fuse2,fuse3),0)

        output = self.conv(x)# acnet

        fuse = fuse.view(1, 3, 512)
        output = output.view(1,3,512)
        output = torch.cat((fuse,output),0)
        output = output.view(4,4,16,12)
        output = self.quaternion_conv(output)
        output = output.view(512,6)
        output = self.linear_layer(output)
        output = output.t()
        return output


if __name__ == '__main__':
    # nl_block=NLBlockND_cross(in_channels=32,dimension=1)
    # 假设 feature_modality1 和 feature_modality2 是两个模态的特征向量
    feature_modality = torch.randn(3,512)
    fusion = conv_att()
    # 特征融合
    fused_features = fusion(feature_modality)
    # fused_features = nl_block(feature_modality1, feature_modality2)
    print(fused_features.shape)