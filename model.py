import torch
import torch.nn as nn

class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_batchnorm=True):
        super(ConvReLU, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = None

        if self.use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        x = self.relu(x)
        return x

class UpSample2D_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), upsample_rate=2, use_batchnorm=True):
        super(UpSample2D_block, self).__init__()
        self.up = nn.Upsample(scale_factor=upsample_rate)
        self.convReLU1 = ConvReLU(in_channels, out_channels, kernel_size, use_batchnorm)
        self.convReLU2 = ConvReLU(out_channels, out_channels, kernel_size, use_batchnorm)
    
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip:
            x = torch.cat([x] + skip, 1)
        x = self.convReLU1(x)
        x = self.convReLU2(x)
        return x

class NestedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        decoder_filters = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2)

        '''
        Backbone: VGG16
        '''
        class VGG16_block1(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, scale_factor=2):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
                self.relu1 = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
                self.relu2 = nn.ReLU(inplace=True)
                self.pool = nn.MaxPool2d(scale_factor)
            
            def forward(self, x):
                x = self.relu1(self.conv1(x))
                x = self.relu2(self.conv2(x))
                return x, self.pool(x)

        class VGG16_block2(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, scale_factor=2):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
                self.relu1 = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
                self.relu2 = nn.ReLU(inplace=True)
                self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
                self.relu3 = nn.ReLU(inplace=True)
                self.pool = nn.MaxPool2d(scale_factor)
            
            def forward(self, x):
                x = self.relu1(self.conv1(x))
                x = self.relu2(self.conv2(x))
                x = self.relu3(self.conv3(x))
                return x, self.pool(x)

        self.vgg16_block1 = VGG16_block1(1, decoder_filters[0])
        self.vgg16_block2 = VGG16_block1(decoder_filters[0], decoder_filters[1])
        self.vgg16_block3 = VGG16_block2(decoder_filters[1], decoder_filters[2])
        self.vgg16_block4 = VGG16_block2(decoder_filters[2], decoder_filters[3])
        self.vgg16_block5 = VGG16_block2(decoder_filters[3], decoder_filters[4])

        '''
        Architecture: U-Net++
        '''
        self.conv0_1 = UpSample2D_block(decoder_filters[0], decoder_filters[0], (3,3), 2, True)
        self.conv1_1 = UpSample2D_block(decoder_filters[1], decoder_filters[1], (3,3), 2, True)
        self.conv2_1 = UpSample2D_block(decoder_filters[2], decoder_filters[2], (3,3), 2, True)
        self.conv3_1 = UpSample2D_block(decoder_filters[3], decoder_filters[3], (3,3), 2, True)

        self.conv0_2 = UpSample2D_block(decoder_filters[0], decoder_filters[0], (3,3), 2, True)
        self.conv1_2 = UpSample2D_block(decoder_filters[1], decoder_filters[1], (3,3), 2, True)
        self.conv2_2 = UpSample2D_block(decoder_filters[2], decoder_filters[2], (3,3), 2, True)

        self.conv0_3 = UpSample2D_block(decoder_filters[0], decoder_filters[0], (3,3), 2, True)
        self.conv1_3 = UpSample2D_block(decoder_filters[1], decoder_filters[1], (3,3), 2, True)

        self.conv0_4 = UpSample2D_block(decoder_filters[0], decoder_filters[0], (3,3), 2, True)

        self.final1 = nn.Conv2d(decoder_filters[1], 1, 1, padding=0)
        self.final2 = nn.Conv2d(decoder_filters[1], 1, 1, padding=0)
        self.final3 = nn.Conv2d(decoder_filters[1], 1, 1, padding=0)
        self.final4 = nn.Conv2d(decoder_filters[1], 1, 1, padding=0)
        self.sigmoid = torch.Sigmoid()

    def forward(self, input):
        '''
        Backbone
        '''
        x0_0, x0_0p = self.vgg16_block1(input)
        x1_0, x1_0p = self.vgg16_block2(x0_0p)
        x2_0, x2_0p = self.vgg16_block3(x1_0p)
        x3_0, x3_0p = self.vgg16_block4(x2_0p)
        x4_0, x4_0p = self.vgg16_block5(x3_0p)

        '''
        U-Net++
        '''
        x0_1 = self.conv0_1(x1_0, x0_0)
        x1_1 = self.conv1_1(x2_0, x1_0)
        x2_1 = self.conv2_1(x3_0, x2_0)
        x3_1 = self.conv3_1(x4_0, x3_0)

        x0_2 = self.conv0_2(x1_1, [x0_0, x0_1])
        x1_2 = self.conv1_2(x2_1, [x1_0, x1_1])
        x2_2 = self.conv2_2(x3_1, [x2_0, x2_1])

        x0_3 = self.conv0_3(x1_2, [x0_0, x0_1, x0_2])
        x1_3 = self.conv1_3(x2_2, [x1_0, x1_1, x1_2])

        x0_4 = self.conv0_4(x1_3, [x0_0, x0_1, x0_2, x0_3])

        output1 = self.sigmoid(self.final1(x0_1))
        output2 = self.sigmoid(self.final2(x0_2))
        output3 = self.sigmoid(self.final3(x0_3))
        output4 = self.sigmoid(self.final4(x0_4))

        return [output1, output2, output3, output4]