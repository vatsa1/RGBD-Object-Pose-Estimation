import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniUNet(nn.Module):
    # TODO: implement a neural network as described in the handout
    def __init__(self):
        """Initialize the layers of the network as instance variables."""
        super(MiniUNet, self).__init__()
        
        # contracting path
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.1)

        # expansive path
        self.conv6 = nn.Conv2d(256+128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128+64, 64, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64+32, 32, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.conv9 = nn.Conv2d(32+16, 16, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(16)
        
        self.conv10 = nn.Conv2d(16, 6, kernel_size=1)
        self.interpolate = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        """
        Forward process.
        """
        x = x.to(dtype=torch.float32)
        
        # Contracting path
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.pool(x1)
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x3 = self.pool(x2)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x4 = self.pool(x3)
        x4 = F.relu(self.bn4(self.conv4(x4)))
        x5 = self.pool(x4)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x5 = self.dropout(x5)

        # Expansive path
        x6 = self.interpolate(x5)
        x7 = torch.cat([x4, x6], dim=1)
        x7 = F.relu(self.bn6(self.conv6(x7)))
        
        x7 = self.interpolate(x7)
        x8 = torch.cat([x3, x7], dim=1)
        x8 = F.relu(self.bn7(self.conv7(x8)))
        
        x8 = self.interpolate(x8)
        x9 = torch.cat([x2, x8], dim=1)
        x9 = F.relu(self.bn8(self.conv8(x9)))
        
        x9 = self.interpolate(x9)
        x10 = torch.cat([x1, x9], dim=1)
        x10 = F.relu(self.bn9(self.conv9(x10)))
        
        output = self.conv10(x10)
        return output


if __name__ == '__main__':
    model = MiniUNet()
    input_tensor = torch.zeros([1, 3, 240, 320])
    output = model(input_tensor)
    print("output size:", output.size())
    print(model)
