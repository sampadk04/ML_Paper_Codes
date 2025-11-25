import numpy as np

from sklearn.metrics import accuracy_score

import torch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt


# helper function to evaluate the generator
def generate_images(generator, nrow, save_path, device):
    # set generator to evaluation mode
    generator.eval()
    
    # create noise_vec
    fixed_noise_batch = torch.randn(size=(nrow*generator.n_classes, generator.noise_size), device=device)

    # create fixed labels (to condition the noise, to generate specific digits)
    fixed_labels = torch.tensor([[i for i in range(generator.n_classes)] for _ in range(nrow)], dtype=torch.int, device=device)

    # generate images from noise and labels
    with torch.no_grad():
        X_gen = generator(fixed_noise_batch, fixed_labels)
        # copy back to cpu
        X_gen = X_gen.cpu()
    
    # normalize the image to bring it from pixel values within (-1,1) to (0,1)
    X_gen = (X_gen + 1)/2
    
    # images in a grid
    image_grid = make_grid(X_gen, nrow)
    # move to last channel
    image_grid = image_grid.permute(1, 2, 0)
    
    # plot images
    plt.figure()
    plt.imshow(image_grid)
    plt.axis(False)
    plt.grid(False)
    # save the images
    # plt.show()
    plt.savefig(save_path)
    plt.close()

# helper function to plot learning curve
def plot_learning_curve(G_Losses, D_Losses, savepath):
    plt.figure()
    plt.plot(G_Losses, label="Generator Losses")
    plt.plot(D_Losses, label="Discriminator Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig(savepath)
    plt.close()