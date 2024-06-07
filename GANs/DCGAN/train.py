import os

import torch

from tqdm import tqdm

# import helper functions
from utils import create_noise, create_ones, create_zeros, generate_images

# setting the device to "mps" instead of default "cpu"
device = torch.device("mps" if torch.backends.mps.is_available else "cpu")

def train_generator(generator, discriminator, X_fake, optimizer, criterion):
    # zero grad
    optimizer.zero_grad()
    
    # extract batch size
    n = X_fake.shape[0]
    
    # create a vector of true labels (w.r.t. Generator)
    # since, the generator is trying to fool the discriminator, the true labels for these generated images should be `1` (w.r.t. generator)
    real_label = create_ones(n)
    
    # predict the labels on the generated samples (generated by the generator) by the discriminator
    predicted_label = discriminator(X_fake)

    # calculate loss
    loss = criterion(predicted_label, real_label)
    # compute grads
    loss.backward()
    # update weights
    optimizer.step()

    return loss


def train_discriminator(generator, discriminator, X_real, X_fake, optimizer, criterion):
    # zero grad
    optimizer.zero_grad()
    
    # extract batch size
    n = X_real.shape[0]
    
    # the discriminator is trying to distinguish between fake (generated images) and real images (images from the dataset)

    # training on the real images from the dataset
    # so the assigned labels for the real images (images from the dataset) should be `1`
    prediction_real = discriminator(X_real)
    # compute real loss
    loss_real = criterion(prediction_real, create_ones(n))
    # compute grad
    loss_real.backward()


    # training on the fake images generated by the generator
    # so the assigned labels for the fake images (images generated by the generator) should be `0`
    prediction_fake = discriminator(X_fake)
    # compute fake loss
    loss_fake = criterion(prediction_fake, create_zeros(n))
    # compute grad
    loss_fake.backward()

    # update the grads
    optimizer.step()

    return loss_real + loss_fake


from sklearn.metrics import accuracy_score

def train_gan(savedir, generator, discriminator, image_dataloaders, optimizers, criterions, num_epochs=150, k=1):
    # keep track of generator and discriminator losses during training
    g_losses = []
    d_losses = []

    # train generator and discriminator
    for epoch in tqdm(range(num_epochs)):
        # keep track of epoch training loss of generator and discriminator
        g_epoch_loss = 0.0
        d_epoch_loss = 0.0  
        
        for phase in ['train', 'test']:
            if phase == 'train':
                
                # extarct training data (both fake and real)
                for batch in image_dataloaders[phase]:
                    # extract features
                    X_real,_ = batch
                    # reshape data and add it to the GPU
                    X_real = X_real.to(device)
                    # extract batch_size
                    batch_size = X_real.shape[0]
                    
                    # create a noise vector to generate inputs
                    noise_vec = create_noise(batch_size, generator.nz)
                    
                    ##############################
                    #   Training discriminator   #
                    ##############################
                    
                    # set disciminator to train mode
                    discriminator.train()
                    # train the discriminator 'k' times
                    for step in range(k):
                        # generate fake images using generator (make sure to not add these to the computational graph), we don't want this included in the gradient calculation
                        # set genertor to eval mode
                        generator.eval()
                        with torch.no_grad():
                            X_fake = generator(noise_vec)
                        d_epoch_loss += train_discriminator(generator, discriminator, X_real, X_fake, optimizers['discriminator'], criterions['discriminator']).item()

                    ##############################
                    #     Training generator     #
                    ##############################
                    
                    # set generator to train mode
                    generator.train()
                    # train the generator once
                    X_fake = generator(noise_vec)
                    # batchnorm is unstable in eval due to generated images change drastically every epoch. So, we won't set discriminator to eval mode here.
                    g_epoch_loss += train_generator(generator, discriminator, X_fake, optimizers['generator'], criterions['generator']).item()

            elif phase=='test':
                # set the generator and discriminator to evaluation mode
                generator.eval()
                # discriminator.eval()
                
                # save images every 20 epochs
                if epoch%10==0:
                    # save img name
                    saveimg_name = 'generated_images_' + str(epoch) + '.png'
                    # check samples generated by the generator
                    generate_images(generator, nrow=8, save_path=os.path.join(savedir, saveimg_name))

                # save model after every 50 epochs
        
        # update the losses
        n_batches = len(image_dataloaders['train'])
        with torch.no_grad():
            g_losses.append(g_epoch_loss/n_batches)
            d_losses.append(d_epoch_loss/(k*n_batches))

        # print training loss info after every 5 epochs
        if epoch%5 == 0:
            print('Epoch {}: Generator Loss: {:.6f} Discriminator Loss: {:.6f}\r'.format(epoch+1, g_losses[-1], d_losses[-1]))

    
    # return generator, discriminator, gen_losses and disc_losses
    return generator, discriminator, g_losses, d_losses