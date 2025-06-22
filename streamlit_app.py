import streamlit as st, torch, torch.nn as nn
from PIL import Image
LATENT = 100

# ---- Generator (identical weights/architecture to training) ----
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 50)
        self.fc = nn.Sequential(nn.Linear(LATENT+50,128*7*7),
                                nn.BatchNorm1d(128*7*7), nn.ReLU(True))
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64,1,4,2,1), nn.Tanh())
    def forward(self,z,y):
        x = torch.cat([z, self.embed(y)], 1)
        x = self.fc(x).view(-1,128,7,7)
        return self.conv(x)

@st.cache_resource
def load_G():
    g = Generator()
    g.load_state_dict(torch.load("cgan_mnist_G.pth", map_location="cpu"))
    g.eval()
    return g

G = load_G()

# ---- Streamlit UI ----
st.set_page_config("Digit Generator", "✍️")
st.title("Hand-written Digit Image Generator")
st.write("Select a digit, click *Generate Images*, get five fresh MNIST-style pictures.")

digit = st.selectbox("Digit", list(range(10)), 0)
if st.button("Generate Images"):
    z   = torch.randn(5, LATENT)
    y   = torch.full((5,), int(digit), dtype=torch.long)
    with torch.no_grad():
        imgs = G(z, y).add(1).div(2).mul(255).byte()   # scale to [0,255]

    st.subheader(f"Generated images of digit **{digit}**")
    cols = st.columns(5)
    for i,(col,img) in enumerate(zip(cols, imgs)):
        pil = Image.fromarray(img.squeeze(0).numpy(), mode="L")
        col.image(pil, use_column_width=True, caption=f"Sample {i+1}")
