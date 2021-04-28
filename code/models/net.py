import torch
import torch.nn as nn
from models import modules
from models.refinenet_dict import refinenet_dict

from models import block as B


def cubes_2_maps(cubes):
    b, c, d, h, w = cubes.shape
    cubes = cubes.permute(0, 2, 1, 3, 4)

    return cubes.contiguous().view(b*d, c, h, w), b, d


def maps_2_cubes(maps, b, d):
    bd, c, h, w = maps.shape
    cubes = maps.contiguous().view(b, d, c, h, w)

    return cubes.permute(0, 2, 1, 3, 4)


def rev_maps(maps, b, d):
    """reverse maps temporarily."""
    cubes = maps_2_cubes(maps, b, d).flip(dims=[2])

    return cubes_2_maps(cubes)[0]


class CoarseNet(nn.Module):
    def __init__(self, encoder, decoder, block_channel, refinenet, bidirectional=False, input_residue=False):

        super(CoarseNet, self).__init__()

        self.use_bidirect = bidirectional
        self.input_residue = input_residue

        self.E = encoder
        self.D = decoder
        self.MFF = modules.MFF(block_channel)

        self.R_fwd = refinenet_dict[refinenet](block_channel)

        if self.use_bidirect:
            self.R_bwd = refinenet_dict[refinenet](block_channel)
            self.bidirection_fusion = nn.Conv2d(6, 3, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, x):
        x_cube = x

        x, b, d = cubes_2_maps(x)
        x_block0, x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block0, x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)])
        fwd_out = self.R_fwd(torch.cat((x_decoder, x_mff), 1), b, d)

        if self.use_bidirect:
            bwd_out = self.R_bwd(torch.cat((rev_maps(x_decoder, b, d),
                                            rev_maps(x_mff, b, d)), 1),
                                 b, d)

            concat_cube = cubes_2_maps(torch.cat((fwd_out, bwd_out.flip(dims=[2])), 1))[0]

            out = maps_2_cubes(self.bidirection_fusion(concat_cube), b=b, d=d)
        else:
            out = fwd_out

        # resolve odd number pixels
        if x_cube.shape[-1] % 2 == 1:
            out = out[:, :, :, :, :-1]

        if self.input_residue:
            return out + x_cube
        else:
            return out


class FineNet(nn.Module):
    def __init__(self, num_features=64, num_blocks=9, out_nc=3, mode='CNA', act_type='relu', norm_type=None):
        super(FineNet, self).__init__()

        nb, nf = num_blocks, num_features

        # 3d convolution to fuse sequence.
        c3d = C_C3D_1()

        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=norm_type, act_type=None)

        self.net = B.sequential(c3d, B.ShortcutBlock_ZKH(B.sequential(*rb_blocks), LR_conv), HR_conv0, HR_conv1)

    def forward(self, c_out, c_in):
        # 6-channel = 3 (rgb) model_out_C + 3 (rgb) in_videos_C
        inp_6c_F = torch.cat((c_out, c_in), 1)
        return self.net(inp_6c_F) + c_out[:, :, c_out.shape[2] // 2, :, :]


class FineNet_npic(nn.Module):
    def __init__(self, num_features=64, num_blocks=9, out_nc=15, mode='CNA', act_type='relu', norm_type=None):
        super(FineNet_npic, self).__init__()

        nb, nf = num_blocks, num_features

        # 3d convolution to fuse sequence.
        c3d = C_C3D_1()

        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=norm_type, act_type=None)

        self.net = B.sequential(c3d, B.ShortcutBlock_ZKH(B.sequential(*rb_blocks), LR_conv), HR_conv0, HR_conv1)

    def forward(self, c_out, c_in):
        b, c, d, h, w = c_out.shape

        # 6-channel = 3 (rgb) model_out_C + 3 (rgb) in_videos_C
        inp_6c_F = torch.cat((c_out, c_in), 1)

        return self.net(inp_6c_F).reshape(b, c, d, h, w) + c_out
        # return self.net(inp_6c_F).reshape(b, c, d, h, w) + c_in


class C_C3D_1(nn.Module):

    def __init__(self, out_channels=64, input_channels=6):
        self.inplanes = out_channels
        super(C_C3D_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, out_channels, kernel_size=5, stride=1, padding=[1, 2, 2], bias=False),
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=[0, 1, 1], bias=False),
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        x = self.conv4(x) + x

        x = x.squeeze(2)

        return x


if __name__ == '__main__':
    s = b, c, d, h, w = 1, 6, 5, 224, 224
    t = torch.ones(s).cuda()

    net = FineNet().cuda()
    output = net(t)
    import pdb; pdb.set_trace()


