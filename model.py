import torch
import torch.nn as nn


# NetG - Generator
class NetG(nn.Module):
    '''
    n_z  : length of the input noise
    n_l  : lenght of the latent representations
    n_t  : length of the skip-thought vector
    n_c  : number of feature maps after first conv layer
    '''

    def __init__(self, n_z=100, n_l=100, n_t=4800, n_c=64):
        super(NetG, self).__init__()
        self.n_z = n_z
        self.n_l = n_l
        self.n_t = n_t
        self.n_c = n_c

        # state size: 8 x 8
        self.convtr1 = nn.ConvTranspose2d(in_channels=n_c*8, out_channels=n_c*4, kernel_size=4, stride=2, padding=1, bias=True)
        self.convtr1_bn = nn.BatchNorm2d(num_features=n_c*4)
        # state size: 16 x 16
        self.convtr2 = nn.ConvTranspose2d(in_channels=n_c*4, out_channels=n_c*2, kernel_size=4, stride=2, padding=1, bias=True)
        self.convtr2_bn = nn.BatchNorm2d(num_features=n_c*2)
        # state size: 32 x 32
        self.convtr3 = nn.ConvTranspose2d(in_channels=n_c*2, out_channels=n_c, kernel_size=4, stride=2, padding=1, bias=True)
        self.convtr3_bn = nn.BatchNorm2d(num_features=n_c)
        # state size: 64 x 64
        self.convtr4 = nn.ConvTranspose2d(in_channels=n_c, out_channels=3, kernel_size=4, stride=2, padding=1, bias=True)
        #self.convtr4_bn = nn.BatchNorm2d(num_features=3) #! user batch normalization for the last layer?
        # state size: 128 x 128 
        
        # linear transformation for the skip-thought vector
        self.lin_emb = nn.Linear(in_features=n_t, out_features=n_l)
        # linear transformation for the concatenated noise
        self.lin_zc = nn.Linear(in_features=n_z+n_l, out_features=8*8*(8*n_c))
        # funcationals
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        # initialize the weights
        self.intialize_weights_()
        
    def forward(self, noise, skip_v):
        emb = self.ReLU(self.lin_emb(skip_v)) 
        z_c = torch.cat([noise,emb],1) # concatenate the noise and embedding
        x = self.ReLU(self.lin_zc(z_c)) #! use ReLU or Leaky ReLU?
        x = x.view(x.size(0), 8*self.n_c, 8, 8) # state size: (8*n_c) x 8 x 8
        x = self.ReLU(self.convtr1_bn(self.convtr1(x)))
        x = self.ReLU(self.convtr2_bn(self.convtr2(x)))
        x = self.ReLU(self.convtr3_bn(self.convtr3(x)))
        x = self.Tanh(self.convtr4(x))
        x = (x/2.0) + 0.5
        return x

    def intialize_weights_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# Discriminator Network - D(x)
class NetD(nn.Module):
    '''
    n_cls : number of classes in the dataset
    n_t   : lenght of the embedding after applying the transformation
    n_f   : number of filters in the first convolutional layer
    m_d   : size of the image before concatenation with the embedding
    '''
    
    def __init__(self, n_cls=102, n_t=256, n_f=64, m_d=8):
        super(NetD, self).__init__()
        
        self.n_f = n_f
        self.n_cls = n_cls
        self.n_t = n_t
        self.m_d = m_d
        # state size: 128 x 128
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_f, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv1_bn = nn.BatchNorm2d(n_f)
        # state size: 64 x 64
        self.conv2 = nn.Conv2d(in_channels=n_f, out_channels=n_f*2, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv2_bn = nn.BatchNorm2d(n_f*2)
        # state size: 32 x 32
        self.conv3 = nn.Conv2d(in_channels=n_f*2, out_channels=n_f*4, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv3_bn = nn.BatchNorm2d(n_f*4)
        # state size: 16 x 16
        self.conv4 = nn.Conv2d(in_channels=n_f*4, out_channels=n_f*6, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv4_bn = nn.BatchNorm2d(n_f*6)
        # state size: 8 x 8
        self.conv5 = nn.Conv2d(in_channels=2*n_f*6, out_channels=n_f*8, kernel_size=1, stride=1, padding=0, bias=True)
        # state size: 8 x 8

        # linear transformation for the skip-thought vector
        self.fc_emb = nn.Linear(in_features=4800, out_features=n_t)
        self.fc_t = nn.Linear(in_features=m_d*m_d*n_f*8, out_features=n_f)
        # linear transformation for the discriminator
        self.fc_d = nn.Linear(in_features=n_f, out_features=1)
        # linear transforation for the classifier
        self.fc_c = nn.Linear(in_features=n_f, out_features=n_cls)
        # functionals
        self.LeakyReLU = nn.LeakyReLU()
        self.Sigmoid = nn.Sigmoid()
        self.LogSoftmax = nn.LogSoftmax()
        # intialize the weights
        self.intialize_weights_()

    def forward(self, input, skip_v):
        x = self.LeakyReLU(self.conv1_bn(self.conv1(input)))
        x = self.LeakyReLU(self.conv2_bn(self.conv2(x)))
        x = self.LeakyReLU(self.conv3_bn(self.conv3(x)))
        x = self.LeakyReLU(self.conv4_bn(self.conv4(x)))
        emb = self.LeakyReLU(self.fc_emb(skip_v))
        emb = emb.view(emb.size(0), int(self.n_t/(self.m_d*self.m_d)), self.m_d, self.m_d) # state size: 4* 8x8
        emb = emb.repeat(1,96,1,1) # state size: 384* 4x4 (tiles the reshaped embedding for 96 times)
        x = torch.cat([x,emb], 1)
        x = self.LeakyReLU(self.conv5(x))
        x = x.view(x.size(0), self.m_d*self.m_d*self.n_f*8)
        x = self.LeakyReLU(self.fc_t(x))
        s = self.Sigmoid(self.fc_d(x))
        c = self.LogSoftmax(self.fc_c(x))
        return s,c 

    def intialize_weights_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()