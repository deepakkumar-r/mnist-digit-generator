import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Define the Conditional Variational Autoencoder (CVAE) - same as in training.py
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

# One-hot encoding for labels
def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets

# --- Streamlit App ---

st.set_page_config(layout="wide")

st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained model.")

# Model parameters (must match training)
feature_size = 28 * 28
latent_size = 16
class_size = 10
model_path = 'cvae.pth'

# Load the trained model
@st.cache_resource
def load_model():
    model = CVAE(feature_size, latent_size, class_size)
    try:
        # Load weights onto CPU
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error(f"Model file not found at '{model_path}'. Please train the model and place 'cvae.pth' in the same directory.")
else:
    # User input
    digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))

    if st.button("Generate Images"):
        st.subheader(f"Generated images of digit {digit}")

        with torch.no_grad():
            # Create 5 random latent vectors
            z = torch.randn(5, latent_size)
            
            # Create label tensor
            label = torch.tensor([digit] * 5)
            label_one_hot = one_hot(label, class_size)
            
            # Generate images
            generated_images = model.decode(z, label_one_hot).view(-1, 28, 28)
            
            # Display images
            cols = st.columns(5)
            for i, img_tensor in enumerate(generated_images):
                img_array = (img_tensor.numpy() * 255).astype(np.uint8)
                image = Image.fromarray(img_array)
                with cols[i]:
                    st.image(image, caption=f"Sample {i+1}", width=128) 