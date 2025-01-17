import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, image_size=64, noise_size=100, n_classes=10, embedding_size=28, feature_map_size=32, output_channels=1):
        '''
        image_size: height and width of the image (assume square images)
        noise_size: size of the input noise
        n_classes: no. of output classes
        embedding_size: size of the one-hot encoded label's embeddings
        feature_map_size: size of feature map in generator
        out_channels: no. of output channels of the images generated by the generator
        '''
        super(Generator, self).__init__()
        self.image_size = image_size
        self.noise_size = noise_size
        self.n_classes = n_classes
        self.embed_size = embedding_size
        
        self.ngf = feature_map_size
        self.oc = output_channels

        # define single block
        def block(in_channels, out_channels, kernel_size, stride, padding, normalize=True):
            # put the layers in a list
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            # return the list of layers
            return layers
        
        # define embedding layer: it takes in (N,) (tensor of labels) tensor as input. We embed this to form tensor of size (N, embed_size), because we concatenate this embedding to the input noise of size (N, noise_size) to form an input of size (N, embed_size + noise_size)
        self.embed = nn.Embedding(self.n_classes, self.embed_size)
        # what we actually do: reshape (N, embed_size) as (N, embed_size, 1, 1) and concatenate it with noise which are already being represented as (N, noise_size, 1,1) to make it a tensor of size (N, embed_size + noise_size, 1,1) before feeding it to the generator

        self.generate = nn.Sequential(
            # input: N * (noise_size + embed_size) * 1 * 1
            # (1,1)
            *block(self.noise_size + self.embed_size, self.ngf*16, 4, 1, 0), # (4,4)
            *block(self.ngf*16, self.ngf*8, 4, 2, 1), # (8,8)
            *block(self.ngf*8, self.ngf*4, 4, 2, 1), # (16,16)
            *block(self.ngf*4, self.ngf*2, 4, 2, 1), # (32,32)
            # final layer
            nn.ConvTranspose2d(self.ngf*2, self.oc,  4, 2, 1, bias=False), # (64,64)
            nn.Tanh()
            # output: N * output_channels, img_size, img_size
        )


    def forward(self, noise, labels):
        '''
        noise: (N, noise_size)
        labels: (N,)
        '''
        batch_size = noise.shape[0]

        # reshape (N, noise_size) noise to (N, noise_size, 1, 1)
        noise = noise.view((batch_size, self.noise_size, 1, 1))
        # reshape (N,) labels to (N, embed_size) to (N, embed_size, 1, 1) (unsqueeze twice to increase dimensions twice)
        embedding = self.embed(labels).view((batch_size, self.embed_size, 1, 1))
        
        # concatenate both (along the channels)
        Z = torch.cat([noise, embedding], dim=1)
        # pass through the generator
        out = self.generate(Z)
        # output as (N, out_channels, img_size, img_size)
        return out


class Discriminator(nn.Module):
    def __init__(self, image_size=64, n_classes=10, feature_map_size=32, input_channels=1):
        '''
        image_size: height and width of the image (assume square images)
        n_classes: no. of output classes
        feature_map_size: size of feature map in discriminator
        input_channels: no. of input channels to the discriminator
        '''
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.n_classes = n_classes
        
        self.ndf = feature_map_size
        self.nc = input_channels
        # self.oc = output_channels

        # define single block
        def block(in_channels, out_channels, kernel_size, stride, padding, normalize=True):
            # put all the layers into a list
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            # return the list of layers
            return layers

        # define embedding layer: it takes in (N,) tensor as input (label encoding). We embed this to form tensor of size (N, img_size * img_size), because we concatenate this embedding to the input image tensor of size (N, c, img_size, img_size) to (add an extra channel) to make it (N, c+1, img_size, img_size) befor feeding the discriminator to discriminate
        self.embed = nn.Embedding(self.n_classes, self.image_size * self.image_size)
        
        # define sequence of layers
        self.discriminate = nn.Sequential(
            # input: N * (input_channels + 1) * img_size * img_size
            # (64,64)
            *block(self.nc + 1, self.ndf, 4, 2, 1, normalize=False), # (32,32)
            *block(self.ndf, self.ndf*2, 4, 2, 1), # (16,16)
            *block(self.ndf*2, self.ndf*4, 4, 2, 1), # (8,8)
            *block(self.ndf*4, self.ndf*8, 4, 2, 1), # (4,4)
            # final layer
            nn.Conv2d(self.ndf*8, 1, 4, 1, 0, bias=False), # (1,1)
            # nn.Conv2d(self.ndf*8, self.oc, 4, 1, 0, bias=False), # (1,1)
            nn.Sigmoid()
            # output: N * output_channels * 1 * 1
        )

    def forward(self, images, labels):
        '''
        images: (N, n_channels, img_size, img_size)
        labels: (N,)
        '''
        # reshape to (N, 1, img_size, img_size)
        embedding = self.embed(labels).view(labels.shape[0], 1, self.image_size, self.image_size)
        # concatenate embedding and input (by adding embedding as an extra channel)
        X = torch.cat([images, embedding], dim=1)
        # pass throught the discriminator
        out = self.discriminate(X)
        # output as (N, 1, 1, 1), we reshape it to (N, 1) and then to (N,)
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