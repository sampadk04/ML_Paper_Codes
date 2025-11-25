import os

import argparse

import torch
import torch.nn as nn
from torch.optim import Adam

from data import get_MNIST_dataloader

from networks import Generator, Discriminator, weights_init

from train import train_gan

from utils import plot_learning_curve



if __name__ == '__main__':

    # argument parsers
    parser = argparse.ArgumentParser(description='Conditional GAN on MNIST')
    
    parser.add_argument('--num_epochs', type=int,
    default=151, help="No. of training epochs")
    parser.add_argument('--image_size', type=int, 
    default=64, help="Image Dimension")
    parser.add_argument('--n_classes', type=int, 
    default=10, help="Number of output classes")
    parser.add_argument('--noise_size', type=int, 
    default=100, help="Size of the latent/noise vector")
    parser.add_argument('--embedding_size', type=int, 
    default=28, help="Size of the label embedding in generator")
    parser.add_argument('--feature_map_size_g', type=int, 
    default=16, help="Number of features to be used in Generator network")
    parser.add_argument('--feature_map_size_d', type=int, 
    default=8, help="Number of features to be used in Discriminator network")
    parser.add_argument('--batch_size', type=int, 
    default=32, help="Size of each batch while training")
    parser.add_argument('--lr_g', type=float, 
    default=0.002, help="Learning Rate for Generator")
    parser.add_argument('--lr_d', type=float, 
    default=0.0002, help="Learning Rate for Discriminator")
    parser.add_argument('--data_save_path', type=str, default='./data/', help='Path to download the data')
    parser.add_argument('--image_save_path', type=str, default='./results/', help='Path to save the images')
    parser.add_argument('--model_save_path', type=str, default='./models/', help='Path to save the trained models')

    opt = parser.parse_args()
    print(opt)

    # setting the device to "mps" instead of default "cpu"
    # device = torch.device("mps" if torch.backends.mps.is_available else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    
    # extract MNIST dataloader
    # if running for the first time keep `download=True`
    # image_dataloaders = get_MNIST_dataloader(download=True)
    image_dataloaders = get_MNIST_dataloader(
        image_size=opt.image_size,
        batch_size=opt.batch_size
    )

    # define Generator and Discriminator
    G = Generator(
        image_size=opt.image_size,
        noise_size=opt.noise_size,
        n_classes=opt.n_classes,
        embedding_size=opt.embedding_size,
        feature_map_size=opt.feature_map_size_g,
        output_channels=1
    ).to(device)

    D = Discriminator(
        image_size=opt.image_size,
        n_classes=opt.n_classes,
        feature_map_size=opt.feature_map_size_d,
        input_channels=1
    ).to(device)

    # initialize their weights to follow normal distribution with mean=0 and std=0.02
    G.apply(weights_init)
    D.apply(weights_init)


    # define optimizers
    
    optimizers = {
        'generator': Adam(params=G.parameters(), lr=opt.lr_g, betas=(0.5, 0.999)),
        'discriminator': Adam(params=D.parameters(), lr=opt.lr_d, betas=(0.5, 0.999))
    }
    '''
    optimizers = {
        'generator': Adam(params=G.parameters(), lr=opt.lr_g),
        'discriminator': Adam(params=D.parameters(), lr=opt.lr_d)
    }
    '''
    
    
    # define loss criterions
    criterions = {
        'generator': nn.BCELoss(),
        'discriminator': nn.BCELoss()
    }

    # train model
    generator, discriminator, g_losses, d_losses = train_gan(
        savedir=opt.image_save_path,
        generator=G,
        discriminator=D,
        image_dataloaders=image_dataloaders,
        optimizers=optimizers,
        criterions=criterions,
        num_epochs=opt.num_epochs,
        device=device
    )

    # save the final models
    savepath_g = os.path.join(opt.model_save_path, 'final_generator_' + str(opt.num_epochs) + '.pth')
    savepath_d = os.path.join(opt.model_save_path, 'final_discriminator_' + str(opt.num_epochs) + '.pth')

    torch.save(generator.state_dict(), savepath_g)
    torch.save(discriminator.state_dict(), savepath_d)

    # save the learning curve
    plot_savepath = os.path.join(opt.image_save_path, 'learning_curve.png')
    plot_learning_curve(g_losses, d_losses, plot_savepath)