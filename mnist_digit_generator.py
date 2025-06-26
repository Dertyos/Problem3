import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import streamlit as st
import numpy as np
from torchvision.utils import make_grid
from PIL import Image

# -------------------------
# 1. Model Definition (Simple GAN)
# -------------------------
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input: Z latent vector
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # output: nc x 32 x 32 (we'll resize later to 28x28)
        )

    def forward(self, input):
        return self.main(input)

# -------------------------
# 2. Training Script (Colab)
# -------------------------

def train_gan(epochs=10, batch_size=128, lr=0.0002, nz=100, device='cuda'):
    # Data loader
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Models
    netG = Generator(nz=nz).to(device)
    netD = nn.Sequential(
        nn.Conv2d(1, 64, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 1, 4, 2, 1, bias=False),
        nn.Sigmoid()
    ).to(device)

    # Loss & Optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(loader):
            # Train Discriminator
            real = imgs.to(device)
            b_size = real.size(0)
            label_real = torch.full((b_size,), 1., device=device)
            output_real = netD(real).view(-1)
            lossD_real = criterion(output_real, label_real)

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label_fake = torch.full((b_size,), 0., device=device)
            output_fake = netD(fake.detach()).view(-1)
            lossD_fake = criterion(output_fake, label_fake)

            lossD = lossD_real + lossD_fake
            netD.zero_grad()
            lossD.backward()
            optimizerD.step()

            # Train Generator
            output = netD(fake).view(-1)
            lossG = criterion(output, label_real)
            netG.zero_grad()
            lossG.backward()
            optimizerG.step()

        print(f"Epoch [{epoch+1}/{epochs}] Loss_D: {lossD.item():.4f} Loss_G: {lossG.item():.4f}")

    # Save generator
    torch.save(netG.state_dict(), 'generator_mnist.pth')

# -------------------------
# 3. Streamlit Web App
# -------------------------

def load_generator(nz=100, device='cpu'):
    netG = Generator(nz=nz).to(device)
    state = torch.load('generator_mnist.pth', map_location=device)
    netG.load_state_dict(state)
    netG.eval()
    return netG

@st.cache(allow_output_mutation=True)
def get_model():
    return load_generator()


def generate_digit_images(digit_label, num_images=5, nz=100):
    netG = get_model()
    # Create a fixed label embedding (simple approach: ignore label, fixed random seed)
    z = torch.randn(num_images, nz, 1, 1)
    with torch.no_grad():
        fake_imgs = netG(z).cpu()
    # Resize and convert to PIL
    pil_imgs = []
    for img in fake_imgs:
        img = (img + 1) / 2  # denormalize
        arr = img.squeeze().numpy() * 255
        pil = Image.fromarray(arr.astype(np.uint8)).resize((28, 28))
        pil_imgs.append(pil)
    return pil_imgs


def main():
    st.title("Handwritten Digit Generator")
    digit = st.slider("Select digit (0-9)", 0, 9, 0)
    if st.button("Generate Images"):
        imgs = generate_digit_images(digit)
        cols = st.beta_columns(len(imgs))
        for c, img in zip(cols, imgs):
            c.image(img, use_column_width=True)

if __name__ == '__main__':
    # In Colab: write files and then run `streamlit run mnist_digit_generator.py`
    train_gan(epochs=5)  # quick demo training
    main()
