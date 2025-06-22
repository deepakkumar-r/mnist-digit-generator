import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Define the Conditional Variational Autoencoder (CVAE)
class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_size + class_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_size)
        self.fc_logvar = nn.Linear(256, latent_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size + class_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, feature_size),
            nn.Sigmoid()
        )

    def encode(self, x, y):
        inputs = torch.cat([x, y], 1)
        h1 = self.encoder(inputs)
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        inputs = torch.cat([z, y], 1)
        return self.decoder(inputs)

    def forward(self, x, y):
        mu, logvar = self.encode(x.view(-1, self.feature_size), y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# One-hot encoding for labels
def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets

def main():
    # Training parameters
    batch_size = 128
    epochs = 20 # You can adjust this
    learning_rate = 1e-3
    latent_size = 16
    class_size = 10
    feature_size = 28 * 28

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MNIST dataset
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    model = CVAE(feature_size, latent_size, class_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels_one_hot = one_hot(labels, class_size).to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data, labels_one_hot)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                      f'Loss: {loss.item() / len(data):.4f}')

        print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'cvae.pth')
    print("Model saved to cvae.pth")

if __name__ == '__main__':
    main() 