'''
copyright Alex Whelan 2025
code for building model
'''

import torch

class MindModel(torch.nn.Module):

  def __init__(self, chan_out, z_dim):
    super(MindModel, self).__init__()
    self.kernel_size = 3
    self.chan_out = chan_out
    self.z_dim = z_dim

    self.noise_layers = torch.nn.Sequential(torch.nn.Linear(z_dim, z_dim)
                                            , torch.nn.Linear(z_dim, z_dim)
                                            , torch.nn.Linear(z_dim, z_dim)
                                            , torch.nn.Linear(z_dim, z_dim)
                                            , torch.nn.Linear(z_dim, z_dim)
                                            , torch.nn.Linear(z_dim, z_dim)
                                            , torch.nn.Linear(z_dim, z_dim)
                                            , torch.nn.Linear(z_dim, z_dim)
                                            )

    self.conv1 = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2)
                                      , torch.nn.Conv2d(self.z_dim, self.z_dim / 2, self.kernel_size, stride=1, padding=1)
                                      , torch.nn.BatchNorm2d(self.z_dim / 2)
                                      , torch.nn.LeakyReLU(inplace=True)
                                      )

    self.conv2 = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2)
                                      , torch.nn.Conv2d(self.z_dim / 2, self.z_dim / 4, self.kernel_size, stride=1, padding=1)
                                      , torch.nn.BatchNorm2d(self.z_dim / 4)
                                      , torch.nn.LeakyReLU(inplace=True)
                                      )

    self.conv3 = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2)
                                      , torch.nn.Conv2d(self.z_dim / 4, self.z_dim / 8, self.kernel_size, stride=1, padding=1)
                                      , torch.nn.BatchNorm2d(self.z_dim / 8)
                                      , torch.nn.LeakyReLU(inplace=True)
                                      )
    
    self.conv4 = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2)
                                      , torch.nn.Conv2d(self.z_dim / 8, self.z_dim / 16, self.kernel_size, stride=1, padding=1)
                                      , torch.nn.BatchNorm2d(self.z_dim / 16)
                                      , torch.nn.LeakyReLU(inplace=True)
                                      )

    self.out = torch.nn.Sequential(torch.nn.Conv2d(self.z_dim / 16, self.chan_out, self.kernel_size, stride=1, padding=1)
                                    , torch.nn.Tanh()
                                      )

  def forward(self, x):
    # noise mapping
    x = self.noise_layers(x)

    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)


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