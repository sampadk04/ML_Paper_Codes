import numpy as np

from sklearn.metrics import accuracy_score

import torch
from torch.autograd.variable import Variable
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

# setting the device to "mps" instead of default "cpu"
device = torch.device("mps" if torch.backends.mps.is_available else "cpu")


# helper function to create noise
def create_noise(n_samples, noise_size=128):
    return Variable(torch.randn(n_samples, noise_size, 1, 1)).to(device)

# helper function to create ones array
def create_ones(n_samples):
    return Variable(torch.ones(n_samples)).to(device)

# helper function to create zeros array
def create_zeros(n_samples):
    return Variable(torch.zeros(n_samples)).to(device)

# helper function to evaluate the generator
def generate_images(generator, nrow, save_path):
    # create noise_vec
    noise_vec = create_noise(64, generator.nz)

    # generate images from noise
    with torch.no_grad():
        generated_images = generator(noise_vec)
        generated_images = generated_images.cpu()
    
    # normalize the image to bring it from (-1,1) range to (0,1)
    generated_images = (generated_images + 1)/2
    
    # images in a grid
    image_grid = make_grid(generated_images, nrow)
    # move to last channel
    image_grid = image_grid.permute(1, 2, 0)
    
    # plot images
    plt.figure()
    plt.imshow(image_grid)
    plt.axis(False)
    plt.grid(False)
    # save the images
    plt.savefig(save_path)
    plt.close()

# helper function to plot learning curve
def plot_learning_curve(g_losses, d_losses, savepath):
    plt.figure()
    plt.plot(g_losses, label="Generator Losses")
    plt.plot(d_losses, label="Discriminator Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig(savepath)
    plt.close()