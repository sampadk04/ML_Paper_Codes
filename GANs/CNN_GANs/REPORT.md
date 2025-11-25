# **CNN GANs — Project Summary**

*A consolidated summary of the architecture, concepts, methods, and experiments performed for Vanilla GAN, DCGAN, and CGAN.*

---

## **1. Introduction**

This report documents the implementation and analysis of three Generative Adversarial Network (GAN) architectures:
1.  **Vanilla GAN**: The original GAN architecture using fully connected layers.
2.  **DCGAN (Deep Convolutional GAN)**: An improvement over Vanilla GAN using convolutional layers for better image quality and stability.
3.  **CGAN (Conditional GAN)**: An extension of GANs that conditions the generation process on class labels, allowing for controlled image generation.

All models were trained and tested on the **MNIST** dataset.

---

## **2. Vanilla GAN**

### **2.1 Overview**
The Vanilla GAN is the simplest form of GAN, consisting of two neural networks (Generator and Discriminator) that are Multilayer Perceptrons (MLPs).

### **2.2 Architecture**
*   **Generator**:
    *   Input: Noise vector ($z$) of size 128.
    *   Hidden Layers: 3 Linear layers with LeakyReLU activation.
    *   Output: Linear layer with Tanh activation, producing a flattened 28x28 image (784 dimensions).
*   **Discriminator**:
    *   Input: Flattened image (784 dimensions).
    *   Hidden Layers: 3 Linear layers with LeakyReLU and Dropout (0.3).
    *   Output: Linear layer with Sigmoid activation (probability of real/fake).

### **2.3 Training Details**
*   **Dataset**: MNIST (28x28 grayscale images).
*   **Loss Function**: Standard Minimax Loss (Binary Cross Entropy).
*   **Optimizer**: Adam (lr=2e-4).
*   **Epochs**: 250.
*   **k step**: 1 (Discriminator trained once per Generator update).

### **2.4 Observations**
*   Simple to implement but prone to mode collapse.
*   Generated images can be blurry compared to convolutional approaches.
*   Fully connected layers ignore spatial correlations in images.

---

## **3. DCGAN (Deep Convolutional GAN)**

### **3.1 Overview**
DCGAN replaces the fully connected layers of Vanilla GAN with convolutional layers. This allows the model to learn spatial hierarchies and generate sharper, more realistic images.

### **3.2 Architecture**
*   **Generator**:
    *   Uses **ConvTranspose2d** (fractionally-strided convolutions) to upsample the noise vector.
    *   **Batch Normalization** is applied after every layer except the output.
    *   **ReLU** activation for all layers except the output (Tanh).
    *   Upsamples from latent vector to 28x28 image.
*   **Discriminator**:
    *   Uses **Conv2d** (strided convolutions) to downsample the image.
    *   **Batch Normalization** applied after every layer except the input.
    *   **LeakyReLU** activation (slope 0.2).
    *   Output is a single probability score (Sigmoid).

### **3.3 Training Details**
*   **Dataset**: MNIST (28x28 grayscale images).
*   **Loss Function**: Standard Minimax Loss (Binary Cross Entropy).
*   **Optimizer**: Adam.
*   **Epochs**: 150.
*   **Weights Initialization**: Normal distribution (mean=0, std=0.02).

### **3.4 Key Improvements**
*   **Spatial Awareness**: Convolutional layers capture spatial features better than MLPs.
*   **Stability**: Batch Normalization helps stabilize training.
*   **Quality**: Produces sharper and more coherent digits than Vanilla GAN.

---

## **4. CGAN (Conditional GAN)**

### **4.1 Overview**
CGAN extends the GAN framework by feeding extra information (class labels) to both the Generator and Discriminator. This allows the model to generate images of a specific class (e.g., generating a specific digit '7').

### **4.2 Architecture**
*   **Generator**:
    *   **Input**: Concatenation of Noise vector ($z$) and **Label Embedding**.
    *   **Structure**: Similar to DCGAN (ConvTranspose2d layers), but deeper to handle 64x64 images.
    *   **Conditioning**: The label embedding is reshaped and concatenated with the noise input.
*   **Discriminator**:
    *   **Input**: Concatenation of Image and **Label Embedding** (as an extra channel).
    *   **Structure**: Similar to DCGAN (Conv2d layers).
    *   **Conditioning**: The label embedding is expanded to match image dimensions and concatenated as an additional channel.

### **4.3 Training Details**
*   **Dataset**: MNIST (Resized to **64x64**).
*   **Loss Function**: Standard Minimax Loss (Binary Cross Entropy).
*   **Optimizer**: Adam (lr_g=0.002, lr_d=0.0002).
*   **Epochs**: 151.

### **4.4 Key Features**
*   **Controllability**: Can generate specific digits on demand by providing the desired label.
*   **Higher Resolution**: Implemented to handle 64x64 images, showing capability to scale up.

---

## **5. Comparison**

| Feature | Vanilla GAN | DCGAN | CGAN |
| :--- | :--- | :--- | :--- |
| **Architecture** | MLP (Fully Connected) | CNN (Convolutional) | CNN (Conditional) |
| **Spatial Features** | Ignores spatial structure | Captures spatial hierarchy | Captures spatial hierarchy |
| **Resolution** | 28x28 | 28x28 | 64x64 (Resized) |
| **Conditioning** | Unconditional (Random) | Unconditional (Random) | Conditional (Label-guided) |
| **Stability** | Prone to instability | More stable (BatchNorm) | Stable |
| **Output Quality** | Blurry, noisy | Sharp, realistic | Sharp, specific class |

---

## **6. Conclusion**

This project successfully implemented and compared three fundamental GAN architectures.
*   **Vanilla GAN** served as a baseline, demonstrating the core adversarial concept.
*   **DCGAN** significantly improved image quality by leveraging convolutional networks, proving the importance of spatial feature learning.
*   **CGAN** added a layer of control, allowing for targeted generation of specific digits, which is crucial for real-world applications where user control is required.

The progression from Vanilla GAN to DCGAN and then to CGAN highlights the evolution of generative models towards higher quality, stability, and controllability.
