import torch.nn as nn
import math
import torch

# 
# 
"""
    https://pytorch.org/docs/0.2.0/torchvision/models.html

    https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101

    All pre-trained models expect input images normalized in the same way, 
    i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where 
    H and W are expected to be atleast 224. The images have to be loaded in 
    to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 
    0.406] and std = [0.229, 0.224, 0.225]


class torchvision.transforms.Normalize(mean, std)
    Normalize a tensor image with mean and standard deviation. 
    Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels, 
    this transform will normalize each channel of the input torch.

    *Tensor i.e. input[channel] = (input[channel] - mean[channel]) / std[channel]


How to freeze part of the model
    https://pytorch.org/docs/master/notes/autograd.html
"""




model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x96_96_256 = self.layer1(x)
        x48_48_512 = self.layer2(x96_96_256)
        x24_24_1024 = self.layer3(x48_48_512)
        x12_12_2048 = self.layer4(x24_24_1024)

        return x96_96_256, x48_48_512, x24_24_1024, x12_12_2048


class ResidualConv(nn.Module):
    def __init__(self, in_out_planes = 256):
        super(ResidualConv, self).__init__()

        print("\tResidualConv")

        self.conv1 = nn.Conv2d(in_out_planes, in_out_planes, kernel_size=3, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(in_out_planes)
        
        self.conv2 = nn.Conv2d(in_out_planes, in_out_planes, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(in_out_planes)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Monocular Relative Depth Perception with Web Stereo Data Supervision
        # Figure 4
        """
            Note that, before each residual convolution block, a transitional 
            3 x 3 convolution layer is applied to adjust the channel number of 
            feature maps. More specifically, the channel number of each 
            transitional layer is set to 256 in our experiments.
        """        

        residual = x

        out = self.relu(x)

        out = self.conv1(out)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        
        return out


class BottleNeckConv(nn.Module):
    def __init__(self, in_out_planes = 256):
        super(BottleNeckConv, self).__init__()

        print("\tBottleNeckConv")

        self.conv1 = nn.Conv2d(in_out_planes, in_out_planes / 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_out_planes / 4)
        
        self.conv2 = nn.Conv2d(in_out_planes / 4, in_out_planes / 4, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(in_out_planes / 4)

        self.conv3 = nn.Conv2d(in_out_planes / 4, in_out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_out_planes)
        

        self.conv4 = nn.Conv2d(in_out_planes, in_out_planes / 4, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(in_out_planes / 4)
        
        self.conv5 = nn.Conv2d(in_out_planes / 4, in_out_planes / 4, kernel_size=3, bias=False, padding=1)
        self.bn5 = nn.BatchNorm2d(in_out_planes / 4)

        self.conv6 = nn.Conv2d(in_out_planes / 4, in_out_planes, kernel_size=1, bias=False)
        self.bn6 = nn.BatchNorm2d(in_out_planes)


        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Monocular Relative Depth Perception with Web Stereo Data Supervision
        # Figure 4
        """
            Note that, before each residual convolution block, a transitional 
            3 x 3 convolution layer is applied to adjust the channel number of 
            feature maps. More specifically, the channel number of each 
            transitional layer is set to 256 in our experiments.
        """        

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        out += residual
        
        x = self.relu(out)

        #again
        residual = x

        out = self.conv4(x)
        out = self.bn4(out)
        out = self.relu(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)

        out = self.conv6(out)
        out = self.bn6(out)
        
        out += residual
        
        out = self.relu(out)       

        return out

class FeatureFusion(nn.Module):
    def __init__(self, block, in_left_planes, in_up_planes, inter_planes = 256, out_planes = 256):
        super(FeatureFusion, self).__init__()

        self.conv_trans_left = nn.Conv2d(in_left_planes, inter_planes, 
                                                kernel_size=3, bias=False, padding=1)
        self.bn_trans_left = nn.BatchNorm2d(inter_planes)
        self.resConv_left = block(in_out_planes = inter_planes)

        self.conv_trans_up = nn.Conv2d(in_up_planes, inter_planes, 
                                                kernel_size=3, bias=False, padding=1)
        self.bn_trans_up = nn.BatchNorm2d(inter_planes)

        

        self.resConv_down = block(in_out_planes = out_planes)

    def forward(self, in_left, in_up):
        """
            Monocular Relative Depth Perception with Web Stereo Data Supervision
            Figure 4
        """
        x_left = self.conv_trans_left(in_left)
        x_left = self.bn_trans_left(x_left)        
        x_left = self.resConv_left(x_left)
        

        x_up = self.conv_trans_up(in_up)
        x_up = self.bn_trans_up(x_up)        

        x = x_left + x_up

        x = self.resConv_down(x)
        x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
                       
        return x

class AdaptiveOutput(nn.Module):
    def __init__(self, inplanes):
        super(AdaptiveOutput, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 128, kernel_size=3, bias=True, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(inplanes, 1, kernel_size=3, bias=True, padding=1)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Monocular Relative Depth Perception with Web Stereo Data Supervision
        # Figure 4
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
                       
        return x

class ReDWebNet_resnet50(nn.Module):
    def __init__(self):
        super(ReDWebNet_resnet50, self).__init__()

        print("===================================================")
        print("Using ReDWebNet_resnet50")   
        print("\tLoading pretrained resnet 50...")
        print("==================================================="        )
        # Encoder
        self.resnet_model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3])
        self.resnet_model.load_state_dict(torch.load('resnet50_no_fc.bin'))
        


        # Decoder, from top to bottom as in Figure 4
        self.feafu3 = FeatureFusion(block = ResidualConv, in_left_planes=1024, in_up_planes = 2048, inter_planes = 512, out_planes = 512) # in:24 out:48
        self.feafu2 = FeatureFusion(block = ResidualConv, in_left_planes=512, in_up_planes = 512, inter_planes = 256, out_planes = 256)   # in:48 out 96
        self.feafu1 = FeatureFusion(block = ResidualConv, in_left_planes=256, in_up_planes = 256, inter_planes = 128, out_planes = 128)   # in:96 out:192
                
        self.ada_out = AdaptiveOutput(inplanes=128)


    def forward(self,x):
        # the input resolution is supposed to be 384 x 384        
        grey96_96_256, grey48_48_512, grey24_24_1024, grey12_12_2048 = self.resnet_model(x)

        blue24_24_2048 = nn.UpsamplingBilinear2d(size=grey24_24_1024.size()[2:])(grey12_12_2048)

        blue48_48 = self.feafu3(in_left = grey24_24_1024, in_up = blue24_24_2048)

        blue96_96 = self.feafu2(in_left = grey48_48_512, in_up = blue48_48)

        blue192_192 = self.feafu1(in_left = grey96_96_256, in_up = blue96_96)

        out = self.ada_out(blue192_192)

        return out


    def prediction_from_output(self, outputs):
        return outputs


class ReDWebNet_resnet50_raw(nn.Module):
    def __init__(self):
        super(ReDWebNet_resnet50_raw, self).__init__()

        print("===================================================")
        print("Using ReDWebNet_resnet50_raw")   
        print("\t No pretraining!")
        print("==================================================="        )
        # Encoder
        self.resnet_model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3])
               


        # Decoder, from top to bottom as in Figure 4
        self.feafu3 = FeatureFusion(block = ResidualConv, in_left_planes=1024, in_up_planes = 2048, inter_planes = 512, out_planes = 512) # in:24 out:48
        self.feafu2 = FeatureFusion(block = ResidualConv, in_left_planes=512, in_up_planes = 512, inter_planes = 256, out_planes = 256)   # in:48 out 96
        self.feafu1 = FeatureFusion(block = ResidualConv, in_left_planes=256, in_up_planes = 256, inter_planes = 128, out_planes = 128)   # in:96 out:192
                
        self.ada_out = AdaptiveOutput(inplanes=128)


    def forward(self,x):
        # the input resolution is supposed to be 384 x 384        
        grey96_96_256, grey48_48_512, grey24_24_1024, grey12_12_2048 = self.resnet_model(x)

        blue24_24_2048 = nn.UpsamplingBilinear2d(size=grey24_24_1024.size()[2:])(grey12_12_2048)

        blue48_48 = self.feafu3(in_left = grey24_24_1024, in_up = blue24_24_2048)

        blue96_96 = self.feafu2(in_left = grey48_48_512, in_up = blue48_48)

        blue192_192 = self.feafu1(in_left = grey96_96_256, in_up = blue96_96)

        out = self.ada_out(blue192_192)

        return out


    def prediction_from_output(self, outputs):
        return outputs

class ReDWebNet_resnet101(nn.Module):
    def __init__(self):
        super(ReDWebNet_resnet101, self).__init__()

        print("===================================================")
        print("Using ReDWebNet_resnet101")   
        print("\tLoading pretrained resnet 101...")
        print("==================================================="        )
        # Encoder

        self.resnet_model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3])
        self.resnet_model.load_state_dict(torch.load('resnet101_no_fc.bin'))
        


        # Decoder, from top to bottom as in Figure 4
        self.feafu3 = FeatureFusion(block = BottleNeckConv, in_left_planes=1024, in_up_planes = 2048, inter_planes = 512, out_planes = 512) # in:24 out:48
        self.feafu2 = FeatureFusion(block = BottleNeckConv, in_left_planes=512, in_up_planes = 512, inter_planes = 256, out_planes = 256)   # in:48 out 96
        self.feafu1 = FeatureFusion(block = BottleNeckConv, in_left_planes=256, in_up_planes = 256, inter_planes = 128, out_planes = 128)   # in:96 out:192
                
        self.ada_out = AdaptiveOutput(inplanes=128)


    def forward(self,x):
        # the input resolution is supposed to be 384 x 384        
        grey96_96_256, grey48_48_512, grey24_24_1024, grey12_12_2048 = self.resnet_model(x)

        blue24_24_2048 = nn.UpsamplingBilinear2d(size=grey24_24_1024.size()[2:])(grey12_12_2048)

        blue48_48 = self.feafu3(in_left = grey24_24_1024, in_up = blue24_24_2048)

        blue96_96 = self.feafu2(in_left = grey48_48_512, in_up = blue48_48)

        blue192_192 = self.feafu1(in_left = grey96_96_256, in_up = blue96_96)

        out = self.ada_out(blue192_192)

        return out


    def prediction_from_output(self, outputs):
        return outputs

def resNet_data_preprocess(color_img):
    """

        All pre-trained models expect input images normalized in the same way, 
        i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where 
        H and W are expected to be atleast 224. The images have to be loaded in 
        to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 
        0.406] and std = [0.229, 0.224, 0.225]

        input[channel] = (input[channel] - mean[channel]) / std[channel]
        Args:
            color_img should be a float32 numpy array of 3 x H x W 

    """    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for channel in range(3):
        color_img[channel, :, :] -= mean[channel]
        color_img[channel, :, :] /= std[channel]

    return color_img


if __name__ == "__main__":
    import numpy as np
    from torch.autograd import Variable

    model = ReDWebNet_resnet50().cuda()
    inputs = np.zeros((4,3,240,320),dtype = np.float32)
    inputs = torch.from_numpy(inputs)
    input_var = Variable(inputs.cuda())

    ###### forwarding
    output_var = model(input_var)
    output = output_var.data.cpu()

    print(output.shape)
    print("done")



