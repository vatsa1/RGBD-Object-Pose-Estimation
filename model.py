import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniUNet(nn.Module):
    # TODO: implement a neural network as described in the handout
    def __init__(self):
        """Initialize the layers of the network as instance variables."""
        super(MiniUNet, self).__init__()
        # TODO
        
        #contracting path
        self.conv1 = nn.Conv2d(3, 16, kernel_size= 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.5)

        #expansive path
        self.conv6 = nn.Conv2d(256+128, 128, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(128+64, 64, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(64+32, 32, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(32+16, 16, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(16, 6, kernel_size=1)
        self.interpolate = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        """
        In:
            x: Tensor [batchsize, channel, height, width], channel=3 for rgb input
        Out:
            output: Tensor [batchsize, class, height, width], class=number of objects + 1 for background
        Purpose:
            Forward process. Pass the input x through the layers defined in __init__() to get the output.
        """
        # TODO
        x = x.to( dtype=torch.float32)
        #print(x.shape)
        #x = x.permute(0, 3, 1, 2)
        x1 = F.relu(self.conv1(x)) #3 to 16
        x2 = self.pool(x1)
        x2 = F.relu(self.conv2(x2)) #16 to 32
        x3 = self.pool(x2)
        x3 = F.relu(self.conv3(x3)) #32 to 64
        x4 = self.pool(x3)
        x4 = F.relu(self.conv4(x4)) #64 to 128
        x5 = self.pool(x4)
        x5 = F.relu(self.conv5(x5)) #128 to 256 
        x5 = self.dropout(x5)
        x6 = self.interpolate(x5)
        x7= torch.cat([x4,x6], dim=1)
        x7 = F.relu(self.conv6(x7)) # 256 to 128
        x7 = self.interpolate(x7)
        x8 = torch.cat([x3, x7], dim=1)
        x8 = F.relu(self.conv7(x8))  # 128 to 64
        x8 = self.interpolate(x8)
        x9 = torch.cat([x2, x8], dim=1)
        x9 = F.relu(self.conv8(x9))  # 64 to 32
        x9 = self.interpolate(x9)
        x10 = torch.cat([x1, x9], dim=1)
        x10 = F.relu(self.conv9(x10))  # 32 to 16
        x11 = self.conv10(x10)  # 16 to 6

        output = x11
        return output


if __name__ == '__main__':
    model = MiniUNet()
    input_tensor = torch.zeros([1, 3, 240, 320])
    output = model(input_tensor)
    print("output size:", output.size())
    print(model)
