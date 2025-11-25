import os

import torch

from tqdm import tqdm

# import helper functions
from utils import generate_images


def train_gan(savedir, generator, discriminator, image_dataloaders, optimizers, criterions, device, num_epochs=150):
    # initialize list to store epoch losses for generator and discriminator
    G_Losses = []
    D_Losses = []

    n_batches = len(image_dataloaders['train'])
    
    # establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.0

    print("Starting Training Loop...")
    # for each epoch
    for epoch in tqdm(range(num_epochs)):
        
        # keep track of epoch training loss of generator and discriminator
        g_epoch_loss = 0.0
        d_epoch_loss = 0.0

        for phase in ['train', 'test']:
            ### training phase
            if phase == 'train':
                # for each batch in the training dataloader
                for batch in image_dataloaders[phase]:

                    # extract features, labels from the batch
                    X_real, X_real_labels = batch

                    # extract batch size
                    batch_size = X_real.shape[0]

                    
                    
                    ##############################
                    #   Training discriminator   #
                    ##############################

                    discriminator.train()
                    discriminator.zero_grad()
                    
                    ## Train with all-real batch

                    # format batch
                    X_real = X_real.to(device)
                    X_real_labels = X_real_labels.to(device)

                    # forward pass the real batch (both features, labels) through the discriminator
                    pred_real = discriminator(X_real, X_real_labels).view(-1)

                    # calculate the discriminator loss on all-real batch
                    # current label: 1
                    label = torch.full(size=(batch_size,), fill_value=real_label, dtype=torch.float, device=device)

                    loss_real = criterions['discriminator'](pred_real, label)

                    # calculate gradients for discriminator in backward pass
                    loss_real.backward()

                    ## Train with all-fake batch

                    # generate a batch of latent vectors
                    batch_noise = torch.randn(size=(batch_size, generator.noise_size), device=device)

                    # generate fake images batch with generator
                    X_fake = generator(batch_noise, X_real_labels)

                    # forward pass the fake batch (both, features, labels) through the discriminator
                    pred_fake = discriminator(X_fake.detach(), X_real_labels).view(-1)

                    # calculate the discriminator loss on all-fake batch
                    # current label: 0
                    label.fill_(fake_label)

                    loss_fake = criterions['discriminator'](pred_fake, label)

                    # calculate gradients for this batch for discriminator, accumulated (summed) with previous gradients
                    loss_fake.backward()

                    # store combined loss
                    loss_discriminator = loss_real + loss_fake

                    # update the discriminator
                    optimizers['discriminator'].step()


                    
                    
                    ##############################
                    #     Training generator     #
                    ##############################

                    generator.train()
                    generator.zero_grad()

                    # since, we just updated the discriminator above, we perform another forward pass of all-fake batch through D
                    pred_fake = discriminator(X_fake, X_real_labels).view(-1)

                    # calculate generator loss based on pred_fake
                    # current label: 1
                    label.fill_(real_label) # real_label for generator cost

                    loss_generator = criterions['generator'](pred_fake, label)

                    # calculate gradients for generator in backward pass
                    loss_generator.backward()

                    # update the generator
                    optimizers['generator'].step()


                    
                    
                    
                    # add batch losses to epoch loss
                    with torch.no_grad():
                        d_epoch_loss += loss_discriminator.item()
                        g_epoch_loss += loss_generator.item()


            ### testing phase
            if phase == 'test':

                # append epoch losses to the list
                G_Losses.append(g_epoch_loss/n_batches)
                D_Losses.append(d_epoch_loss/n_batches)

                
                # save generatored images, print stats, save models
                if epoch%10 == 0:
                    # save img name
                    saveimg_name = 'generated_images_' + str(epoch) + '.png'

                    # check samples generated by the generator
                    generate_images(generator, nrow=10, save_path=os.path.join(savedir, saveimg_name), device=device)

                    # save models
                    if epoch%50 == 0:
                        print("Saving Models...")

                        generator_name = 'generator_' + str(epoch) + '.pt'
                        discriminator_name = 'discriminator_' + str(epoch) + '.pt'
                        
                        # save the models every 50 epochs
                        torch.save(generator.state_dict(), os.path.join('models', generator_name))
                        torch.save(discriminator.state_dict(), os.path.join('models', discriminator_name))
                        
                        print("Models Saved!")
                

                
                # print training loss after every 5 epochs
                if epoch%2 == 0:
                    print('Epoch {}: Generator Loss: {:.6f} Discriminator Loss: {:.6f}\r'.format(epoch+1, G_Losses[-1], D_Losses[-1]))
    

    # return generator, discriminator, gen_losses and disc_losses
    return generator, discriminator, G_Losses, D_Losses