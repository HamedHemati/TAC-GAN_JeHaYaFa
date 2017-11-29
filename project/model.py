import torch
import torch.nn as nn


# custom weight normlization
def normal_init(m, mean=0.0, std=0.02):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# Generator Network - G(z)
class NetG(nn.Module):
    # d : number of layers after first conv layer
    def __init__(self, nz, d=128):
        super(NetG, self).__init__()
        self.nz = nz # length of the input noise
        # output size: 64x64
        # Conv-Transpose layers
        self.convtr1 = nn.ConvTranspose2d(in_channels=self.nz, out_channels=d*8, kernel_size=4, stride=1, padding=0)
        self.convtr1_bn = nn.BatchNorm2d(num_features=d*8)
        self.convtr2 = nn.ConvTranspose2d(in_channels=d*8, out_channels=d*4, kernel_size=4, stride=2, padding=1)
        self.convtr2_bn = nn.BatchNorm2d(num_features=d*4)
        self.convtr3 = nn.ConvTranspose2d(in_channels=d*4, out_channels=d*2, kernel_size=4, stride=2, padding=1)
        self.convtr3_bn = nn.BatchNorm2d(num_features=d*2)
        self.convtr4 = nn.ConvTranspose2d(in_channels=d*2, out_channels=d, kernel_size=4, stride=2, padding=1)
        self.convtr4_bn = nn.BatchNorm2d(num_features=d)
        self.convtr5 = nn.ConvTranspose2d(in_channels=d, out_channels=3, kernel_size=4, stride=2, padding=1)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)
        self.Tanh = nn.Tanh()
        self.apply(normal_init)

    def forward(self, x):
        x = self.LeakyReLU(self.convtr1_bn(self.convtr1(x)))
        x = self.LeakyReLU(self.convtr2_bn(self.convtr2(x)))
        x = self.LeakyReLU(self.convtr3_bn(self.convtr3(x)))
        x = self.LeakyReLU(self.convtr4_bn(self.convtr4(x)))
        x = self.Tanh(self.convtr5(x))
        return x


# Discriminator Network - D(x)
class NetD(nn.Module):
    def __init__(self, d=128, num_classes=10):
        super(NetD, self).__init__()
        # input size: 64x64
        self.d = d
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=d, kernel_size=4, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(d)
        self.conv2 = nn.Conv2d(in_channels=d, out_channels=d*2, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(in_channels=d*2, out_channels=d*4, kernel_size=4, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(in_channels=d*4, out_channels=d*8, kernel_size=4, stride=2, padding=1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(in_channels=d*8, out_channels=d, kernel_size=4, stride=1, padding=0)
        self.lin_disc = nn.Linear(in_features=d, out_features=1)
        self.lin_clsf = nn.Linear(in_features=d, out_features=self.num_classes)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)
        self.Sigmoid = nn.Sigmoid()
        self.Softmax = nn.Softmax()
        self.apply(normal_init)

    def forward(self, input):
        x = self.LeakyReLU(self.conv1_bn(self.conv1(input)))
        x = self.LeakyReLU(self.conv2_bn(self.conv2(x)))
        x = self.LeakyReLU(self.conv3_bn(self.conv3(x)))
        x = self.LeakyReLU(self.conv4_bn(self.conv4(x)))
        x = self.LeakyReLU(self.conv5(x))
        x = x.view(x.size(0), self.d)
        s = self.Sigmoid(self.lin_disc(x))
        c = self.Softmax(self.lin_clsf(x))
        return s,c 
