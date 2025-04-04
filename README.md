# Example project for DCGAN 

This example project is a pet practice project based on PyTorch framework. This project allows me to learn and practice the Python core implementation of the GAN network to generate faces of people.

## Components

`generator.py` implemented a generator neural network. `discriminator.py` implemented the discriminator network. `gen-example.ipynb` implemented the application of training and evaluation of generating faces. 

Here is the trained dataset.

![download](https://github.com/user-attachments/assets/252fa4a1-bd40-41e3-b74f-8405e7faa7c1)

Here, Binary Cross Entropy is used to optimize two networks. 
https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
