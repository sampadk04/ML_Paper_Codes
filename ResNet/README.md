# ResNet
We present a comprehensive implementation of the ResNet-18 architecture from scratch using PyTorch. This project is inspired by the seminal paper "Deep Residual Learning for Image Recognition" by Kaiming He et al. Our implementation focuses on replicating the architecture and training it on the CIFAR-10 dataset to achieve robust image classification performance.

## Overview

ResNet-18 is a popular convolutional neural network architecture known for its use of residual learning to effectively train deep networks. In this project, we:

- Implement the ResNet-18 architecture from scratch using PyTorch.
- Train the network on the CIFAR-10 dataset.
- Achieve competitive performance with the original implementation.

<img src="sample_images/ResNet-18-architecture.png" alt="ResNet-18 Architecture" height="200" width="700"/>

## Implementation Details

The implementation is structured as follows:

1. **Architecture Design**: We design the ResNet-18 model, incorporating the key components such as convolutional layers, batch normalization, ReLU activations, and the residual blocks.
2. **Training**: The model is trained on the CIFAR-10 dataset, utilizing data augmentation techniques to improve generalization.
3. **Evaluation**: The performance of the model is evaluated on the test set, demonstrating its effectiveness in image classification tasks.

## Results

Our implementation of ResNet-18 achieves competitive performance on the CIFAR-10 dataset, showcasing the power of residual learning in training deep neural networks. The model demonstrates robustness and generalization capabilities, making it suitable for a wide range of image classification tasks.

<img src="sample_images/results.jpeg" alt="Results" width="500"/>

## Inspiration and References

Our implementation is heavily inspired by the original paper titled "[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385v1)". This paper introduced the concept of residual learning, which has since become a foundational technique in deep learning.

## Conclusion

This project showcases the implementation of ResNet-18 from scratch, demonstrating the power of residual learning in training deep neural networks. By following this implementation, you can gain insights into the inner workings of ResNet-18 and apply similar techniques to your own projects.