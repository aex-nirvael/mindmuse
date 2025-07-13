'''
copyright Alex Whelan 2025
code for building model
'''

import torch

class MindModel(torch.nn.Module):

  def __init__(self, chan_out, z_dim, batch_size):
    super(MindModel, self).__init__()
    self.kernel_size = 3
    self.chan_out = chan_out
    self.dims = [int(z_dim / (2**i)) for i in range(9)]
    self.batch_size = batch_size
    self.z_dim = z_dim
    self.z_features = z_dim * batch_size

    self.noise_layers = torch.nn.Sequential(torch.nn.Linear(self.z_features, self.z_features)
                                            , torch.nn.Linear(self.z_features, self.z_features)
                                            , torch.nn.Linear(self.z_features, self.z_features)
                                            , torch.nn.Linear(self.z_features, self.z_features)
                                            , torch.nn.Linear(self.z_features, self.z_features)
                                            , torch.nn.Linear(self.z_features, self.z_features)
                                            , torch.nn.Linear(self.z_features, self.z_features)
                                            , torch.nn.Linear(self.z_features, self.z_features)
                                            )

    self.conv1 = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2)
                                      , torch.nn.Conv2d(self.dims[0], self.dims[1], self.kernel_size, stride=1, padding=1)
                                      , torch.nn.BatchNorm2d(self.dims[1])
                                      , torch.nn.LeakyReLU(inplace=True)
                                      )

    self.conv2 = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2)
                                      , torch.nn.Conv2d(self.dims[1], self.dims[2], self.kernel_size, stride=1, padding=1)
                                      , torch.nn.BatchNorm2d(self.dims[2])
                                      , torch.nn.LeakyReLU(inplace=True)
                                      )

    self.conv3 = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2)
                                      , torch.nn.Conv2d(self.dims[2], self.dims[3], self.kernel_size, stride=1, padding=1)
                                      , torch.nn.BatchNorm2d(self.dims[3])
                                      , torch.nn.LeakyReLU(inplace=True)
                                      )
    
    self.conv4 = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2)
                                      , torch.nn.Conv2d(self.dims[3], self.dims[4], self.kernel_size, stride=1, padding=1)
                                      , torch.nn.BatchNorm2d(self.dims[4])
                                      , torch.nn.LeakyReLU(inplace=True)
                                      )
    
    self.conv5 = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2)
                                      , torch.nn.Conv2d(self.dims[4], self.dims[5], self.kernel_size, stride=1, padding=1)
                                      , torch.nn.BatchNorm2d(self.dims[5])
                                      , torch.nn.LeakyReLU(inplace=True)
                                      )
    
    self.conv6 = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2)
                                      , torch.nn.Conv2d(self.dims[5], self.dims[6], self.kernel_size, stride=1, padding=1)
                                      , torch.nn.BatchNorm2d(self.dims[6])
                                      , torch.nn.LeakyReLU(inplace=True)
                                      )
    
    self.conv7 = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2)
                                      , torch.nn.Conv2d(self.dims[6], self.dims[7], self.kernel_size, stride=1, padding=1)
                                      , torch.nn.BatchNorm2d(self.dims[7])
                                      , torch.nn.LeakyReLU(inplace=True)
                                      )
    
    self.conv8 = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2)
                                      , torch.nn.Conv2d(self.dims[7], self.dims[8], self.kernel_size, stride=1, padding=1)
                                      , torch.nn.BatchNorm2d(self.dims[8])
                                      , torch.nn.LeakyReLU(inplace=True)
                                      )

    self.out = torch.nn.Sequential(torch.nn.Conv2d(self.dims[8], self.chan_out, self.kernel_size, stride=1, padding=1)
                                    , torch.nn.Tanh()
                                      )

  def forward(self, x):
    # noise mapping
    x = self.noise_layers(x)

    x = x.reshape(self.batch_size, self.z_dim, 1, 1)

    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.conv6(x)
    x = self.conv7(x)
    x = self.conv8(x)

    return self.out(x)
  

class MindDiscriminator(torch.nn.Module):

  def __init__(self):
    super(MindDiscriminator, self).__init__()
    self.kernel_size = 3
    self.chan_in = 3

    self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(self.chan_in, 8, self.kernel_size, stride=2, padding=1)
                                      , torch.nn.BatchNorm2d(8)
                                      , torch.nn.ReLU(inplace=True)
                                      )

    self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(8, 32, self.kernel_size, stride=2, padding=1)
                                      , torch.nn.BatchNorm2d(32)
                                      , torch.nn.ReLU(inplace=True)
                                      )

    self.conv3 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, self.kernel_size, stride=2, padding=1)
                                      , torch.nn.BatchNorm2d(64)
                                      , torch.nn.ReLU(inplace=True)
                                      )

    self.out = torch.nn.Sequential(torch.nn.Conv2d(64, 1, self.kernel_size, stride=1, padding=1)
                                    , torch.nn.Sigmoid()
                                      )

  def forward(self, x):

    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)

    return self.out(x)