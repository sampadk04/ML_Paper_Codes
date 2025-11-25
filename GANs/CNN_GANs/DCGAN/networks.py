import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_size=100, feature_map_size=32, output_channels=1): 
        '''
        noise_size: size of latent vector (generator input)
        feature_map_size: size of feature map in generator
        output_channel: no. of channels output image
        '''
        super(Generator, self).__init__()
        self.nz = noise_size
        self.ngf = feature_map_size
        self.nc = output_channels

        # define the sequence of layers
        self.generate = nn.Sequential(
            # Layer-1: input noise is going into convolution (Input: nz x 1 x 1)
            nn.ConvTranspose2d(in_channels=self.nz, out_channels=self.ngf*4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(inplace=True),

            # Layer-2
            nn.ConvTranspose2d(in_channels=self.ngf*4, out_channels=self.ngf*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(inplace=True),

            # Layer-3
            nn.ConvTranspose2d(in_channels=self.ngf*2, out_channels=self.ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(inplace=True),

            # Layer-4 (Output: nc x 28 x 28)
            nn.ConvTranspose2d(in_channels=self.ngf, out_channels=self.nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, Z):
        out = self.generate(Z)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_channels=1, feature_map_size=32, output_size=1):
        '''
        input_channels: no. of input channels
        feature_map_size: size of feature map in discriminator
        output_size: size of output
        '''
        super(Discriminator, self).__init__()
        self.nc = input_channels
        self.ndf = feature_map_size
        self.oc = output_size

        # define sequence of layers
        self.discriminate = nn.Sequential(
            # Layer-1 (Input: nc x 28 x 28)
            nn.Conv2d(in_channels=self.nc, out_channels=self.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer-2
            nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer-3
            nn.Conv2d(in_channels=self.ndf*2, out_channels=self.ndf*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer-4 (Output: 1 channel)
            nn.Conv2d(in_channels=self.ndf*4, out_channels=self.oc, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, X):
        out = self.discriminate(X)
        return out.view(-1, 1).squeeze(1)


# for custom weight initialization of Generator and Discriminator
# initialize model weights to follow a Normal distribution with mean=0 and std=0.02
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0.0)