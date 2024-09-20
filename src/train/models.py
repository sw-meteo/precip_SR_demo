import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    '''
    - Conv -> BN -> ReLU -> Conv -> BN -> ReLU ->
    in_channels        -> mid channels       -> out_channels
    
    in:     B iC H W
    out:    B oC H W
    '''
    def __init__(self, in_channels, out_channels, mid_channels=None, spectral_norm=False):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        if spectral_norm:
            conv0 = nn.utils.spectral_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False))
            conv1 = nn.utils.spectral_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False))
        else:
            conv0 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
            conv1 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.double_conv = nn.Sequential(
            conv0,
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            conv1,
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UniConv(nn.Module):
    '''
    - Conv -> BN -> ReLU -> 
    in_channels          -> out_channels
    
    in:     B iC H W
    out:    B oC H W
    '''
    def __init__(self, in_channels, out_channels, spectral_norm=False):
        super(UniConv, self).__init__()
        if spectral_norm:
            conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.uni_conv = nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.uni_conv(x)


class Down(nn.Module):
    '''
    - pooling -> DoubleConv ->
    
    in:     B iC H W
    out:    B oC H//2 W//2
    '''
    def __init__(self, in_channels, out_channels, layer='DoubleConv', spectral_norm=False):
        super(Down, self).__init__()
        if layer == 'DoubleConv':
            self.conv = DoubleConv(in_channels, out_channels, spectral_norm=spectral_norm)
        elif layer == 'UniConv':
            self.conv = UniConv(in_channels, out_channels, spectral_norm=spectral_norm)
        else:
            raise ValueError
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            self.conv,
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    '''
    - up (convT) -> DoubleConv ->
    
    in:     B iC H W
    out:    B oC H*2 W*2
    '''
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class CompUp(nn.Module):
    '''
    comp 2 input
    - convT -> cat -> DoubleConv ->
    - padding --^
    
    in: x1: B i1C H W
        x2: B i2C H' W' , typical B iC H*2 W*2
    out: B oC H*2 W*2
        
    process:
        1. x1: B i1C H W  -up->  x1': B i1C//2 H*2 W*2
        2. x2: B i2C H' W' -pad-> x2': B i2C H*2 W*2
        3. concat x1' and x2': B i1C//2+i2C H*2 W*2
        4. DoubleConv to: B oC H*2 W*2
    '''
    def __init__(self, in_channels_1, in_channels_2, out_channels):
        super(CompUp, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels_1, in_channels_1 // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels_1//2+in_channels_2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad x2 to x1
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]
        assert diffY >= 0 and diffX >= 0, \
            'In Up block, diffY and diffX should be non-negative'
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    '''
    Conv1x1
    
    in:     B iC H W
    out:    B oC H W
    '''
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class UNet(nn.Module):
    '''
    An simple image2image model for generator
    '''
    def __init__(self, 
                input_shape=[1, 16, 16],
                output_shape=[1, 64, 64],
                entry_channel=8,
                exit_channel=4,
                out_features=128,
                down_channel_multiplier=[1, 2, 4],
                up_channel_multiplier=[1, 1, 2, 4, 8],
                nz=0,
                *args, **kwargs
                ):
        super(UNet, self).__init__()
        in_channels, in_height, in_width = input_shape
        out_channels, out_height, out_width = output_shape
        self.nz = nz
        
        self.inc = DoubleConv(in_channels,
                              entry_channel * down_channel_multiplier[0])
        self.d1 = Down(entry_channel * down_channel_multiplier[0],
                       entry_channel * down_channel_multiplier[1])
        self.d2 = Down(entry_channel * down_channel_multiplier[1],
                       entry_channel * down_channel_multiplier[2])
        self.link = DoubleConv(entry_channel * down_channel_multiplier[2],
                               exit_channel * up_channel_multiplier[4])
        self.u3 = CompUp((exit_channel * up_channel_multiplier[4] + nz),
                         entry_channel * down_channel_multiplier[1],
                         exit_channel * up_channel_multiplier[3])
        self.u2 = CompUp((exit_channel * up_channel_multiplier[3]),
                         entry_channel * down_channel_multiplier[0],
                         exit_channel * up_channel_multiplier[2])
        self.u1 = Up(exit_channel * up_channel_multiplier[2],
                     exit_channel * up_channel_multiplier[1])
        self.u0 = Up(exit_channel * up_channel_multiplier[1],
                     exit_channel * up_channel_multiplier[0])
        self.outc = OutConv(exit_channel * up_channel_multiplier[0],
                            out_features)
        self.mid_act = nn.LeakyReLU()
        self.outfc = nn.Linear(out_features, out_channels)
        self.last_act = nn.Identity()
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x = self.link(x3)
        if self.nz > 0:
            B, L = z.size()
            _, _, H, W = x.size()
            z = z.view(B, L, 1, 1).expand(-1, -1, H, W)
            x = torch.cat([x, z], dim=1)
        x = self.u3(x, x2)
        x = self.u2(x, x1)
        x = self.u1(x)
        x = self.u0(x)
        x = self.outc(x)
        x = self.mid_act(x)
        x = x.permute(0, 2, 3, 1) # B C H W -> B H W C
        x = self.outfc(x)
        x = x.permute(0, 3, 1, 2) # B H W C -> B C H W
        x = self.last_act(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary
    
    Gmodel = UNet()
    print(Gmodel)
    X = torch.randn(2, 1, 16, 16)
    print('G out shape: ', Gmodel(X).shape)
    
    Gmodel.cuda()
    summary(Gmodel, (2, 1, 16, 16))
