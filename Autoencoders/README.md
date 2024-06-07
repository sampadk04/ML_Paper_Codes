# Autoencoders

## Implementation of AutoEncoders from scratch using PyTorch

We implement 3 types of AutoEncoders on the MNIST handwritten digits dataset:
- Simple AutoEncoder
- De-Noising AutoEncoder
- Variational AutoEncoder

Implementation of Variational AutoEncoder has been inspired from the original AutoEncoder paper named "[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114v10)".

Here are some demonstration from the codes:

- Reconstruction of Simple AutoEncoder:
<img src="sample_images/SAE_reconstruction.jpeg" alt="SAE_reconstruction" width="400"/>

- Reconstruction of De-Noising AutoEncoder:
<img src="sample_images/DAE_reconstruction.jpeg" alt="DAE_reconstruction" width="400"/>

- Interpolation by Variational AutoEncoder:
<img src="sample_images/VAE_interpolation.jpeg" alt="VAE_interpolation" width="400"/>