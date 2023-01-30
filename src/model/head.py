import torch.nn as nn
import torch.nn.functional as F


class mmdetection_head(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, do_downsample=False):
        super(mmdetection_head, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=True, dilation=2)  # dilation
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=True)

        ####################################################################
        # TODO: Please complete the downsample module
        # Hint: Use a "kernel_size=1"'s convolution layer to align the dimension
        # Hint: We don't suggest using any batch normalization on detection head.
        #####################################################################
        self.downsample = nn.Sequential()
        if do_downsample or stride != 1 or in_planes != self.expansion * planes:
            pass

        ##################################################################

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out += self.downsample(x)
        out = F.relu(out)
        return out
