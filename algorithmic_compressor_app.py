import streamlit as st
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import io, time, json, os

# Optional libs
try:
    import zstandard as zstd
    HAS_ZSTD = True
except Exception:
    import zlib
    HAS_ZSTD = False

try:
    from pysr import PySRRegressor
    HAS_PYSR = True
except Exception:
    HAS_PYSR = False

from scipy.optimize import curve_fit

st.set_page_config(layout="wide")
st.title("Algorithmic Compressor — Research Prototype")
st.markdown(
    '<p style="font-size:14px; color:gray;">Source code: <a href="https://github.com/TruongLouis-Programmer/Algorithmic-Compressor-Research-Prototype/blob/main/algorithmic_compressor_app.py" target="_blank">GitHub</a></p>',
    unsafe_allow_html=True
)

# ----------------------
# Utility functions
# ----------------------
def synth_sine(x, A, B, C):
    return A * np.sin(B * x + C)

def synth_linear(x, a, b):
    return a * x + b

def synth_quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def try_curvefit(x, y):
    """Attempt curve fits for common families and return best fit (name, params, mse)"""
    results = []
    try:
        p, _ = curve_fit(synth_sine, x, y, maxfev=2000)
        mse = np.mean((synth_sine(x, *p) - y)**2)
        results.append(("sine", p, mse))
    except Exception:
        pass
    try:
        p, _ = curve_fit(synth_linear, x, y, maxfev=2000)
        mse = np.mean((synth_linear(x, *p) - y)**2)
        results.append(("linear", p, mse))
    except Exception:
        pass
    try:
        p, _ = curve_fit(synth_quadratic, x, y, maxfev=2000)
        mse = np.mean((synth_quadratic(x, *p) - y)**2)
        results.append(("quadratic", p, mse))
    except Exception:
        pass
    if results:
        results.sort(key=lambda t: t[2])
        return results[0] 
    return None

def compress_bytes(b: bytes, level=3):
    if HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=level)
        return cctx.compress(b)
    else:
        return zlib.compress(b, level)

def decompress_bytes(b: bytes):
    if HAS_ZSTD:
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(b)
    else:
        return zlib.decompress(b)

# ----------------------
# Dataset + generators
# ----------------------
def gen_sequence(kind, length=50, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.linspace(0, 1, length)
    if kind == "linear":
        a = np.random.uniform(-3, 3); b = np.random.uniform(-1, 1)
        y = a * x + b
    elif kind == "quadratic":
        a, b, c = np.random.uniform(-2, 2, 3)
        y = a * x ** 2 + b * x + c
    elif kind == "sine":
        A = np.random.uniform(0.5, 2); B = np.random.uniform(1, 5); C = np.random.uniform(0, 2 * np.pi)
        y = A * np.sin(B * x + C)
    elif kind == "fib":
        seq = [0.0, 1.0]
        for _ in range(length - 2):
            seq.append(seq[-1] + seq[-2])
        y = np.array(seq, dtype=float)
        if y.max() != 0:
            y = y / float(y.max())
    else:  
        y = np.random.normal(0, 1, length)
    return y

def make_dataset(n_samples=512, length=50):
    choices = ["sine", "linear", "quadratic", "fib", "noise"]
    probs = [0.25, 0.2, 0.2, 0.15, 0.2]
    X = []
    meta = []
    for _ in range(n_samples):
        k = np.random.choice(choices, p=probs)
        X.append(gen_sequence(k, length))
        meta.append(k)
    return np.array(X, dtype=float), meta

# ----------------------
# Model
# ----------------------
class Compressor(nn.Module):
    def __init__(self, length=50, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(length, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, length)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# ----------------------
# UI controls
# ----------------------
st.sidebar.header("Controls")
seed = st.sidebar.number_input("Random seed", value=42, step=1)
np.random.seed(seed)

example_kind = st.sidebar.selectbox("Example sequence type", ["sine", "linear", "quadratic", "fib", "noise"])
length = st.sidebar.slider("Sequence length", 20, 1000, 1000)
latent_dim = st.sidebar.slider("Latent dimension (z size)", 2, 64, 8)
train_epochs = st.sidebar.slider("Training epochs (quick demo)", 10, 10000, 10000)
use_gpu = st.sidebar.checkbox("Use GPU if available", value=False)
run_train = st.sidebar.button("Run Training & Analysis (demo)")
save_model_btn = st.sidebar.button("Save model to disk")
load_model_btn = st.sidebar.button("Load model from disk")

st.write("## Example input sequence")
st.markdown("This graph shows an example of a single sequence that we will feed into our compressor model. The goal is to compress this sequence into a small 'latent vector' and then reconstruct it back to its original form with minimal loss of information.")
example_seq = gen_sequence(example_kind, length, seed=seed+1)
fig, ax = plt.subplots()
ax.plot(example_seq, label="example")
ax.set_title(f"Example: {example_kind} (length={length})")
ax.legend()
st.pyplot(fig)

st.write("---")
st.write("### Dataset (synthetic)")
st.write("We'll train on a synthetic dataset of mixed algorithmic sequences (sine, linear, quadratic, fib, noise).")

# Paths
MODEL_PATH = "model_v2.pth"

if run_train:
    start_time = time.time()
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    st.info(f"Using device: {device}")

    with st.spinner("Generating dataset..."):
        X, meta = make_dataset(n_samples=512, length=length)
        st.write("Dataset shape:", X.shape)

    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    model = Compressor(length=length, latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    batch_size = 64
    losses = []
    st.write("Training...")
    progress_bar = st.progress(0)
    for epoch in range(train_epochs):
        perm = torch.randperm(X_t.size(0))
        epoch_loss = 0.0
        model.train()
        for i in range(0, X_t.size(0), batch_size):
            batch = X_t[perm[i:i + batch_size]]
            recon, z = model(batch)
            loss = loss_fn(recon, batch) + 1e-3 * torch.mean(z ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= X_t.size(0)
        losses.append(epoch_loss)
        progress_bar.progress((epoch + 1) / train_epochs)
        if (epoch + 1) % max(1, (train_epochs // 5)) == 0:
            st.write(f"Epoch {epoch + 1}/{train_epochs} - Loss {epoch_loss:.6f}")

    st.success("Training finished.")
    duration = time.time() - start_time
    st.write(f"Total time: {duration:.1f}s")

    # Loss curve
    st.markdown("---")
    st.markdown("### Model Performance")
    st.markdown("The **loss curve** below illustrates how the model's error decreased during training. A lower loss value means the model is getting better at reconstructing the original sequences. We want to see this curve go down and flatten out, which indicates the model has learned effectively.")
    fig2, ax2 = plt.subplots()
    ax2.plot(losses)
    ax2.set_title("Training loss curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    st.pyplot(fig2)

    model.eval()
    zs = []
    recon_all = []
    with torch.no_grad():
        for i in range(X_t.size(0)):
            inp = X_t[i:i+1]
            recon, z = model(inp)
            zs.append(z.cpu().numpy().reshape(-1))
            recon_all.append(recon.cpu().numpy().reshape(-1))
    zs = np.array(zs)
    recon_all = np.array(recon_all)

    st.markdown("---")
    st.markdown("### Latent Space Visualization")
    st.markdown("""
    The graphs below help us understand what the model has learned. The model compresses each sequence into a small set of numbers called a **latent vector (z)**. This vector is a compressed representation of the original sequence.

    We use techniques like **PCA** and **t-SNE** to visualize these high-dimensional latent vectors in 2D. 
    
    **What to look for:** If the model has learned well, sequences generated by the same algorithm (e.g., all 'sine' waves) should have similar latent vectors. In the graph, this will appear as distinct **clusters** of points for each sequence type. This shows the model has discovered the underlying 'algorithmic family' of the data.
    """)
    scaler = StandardScaler()
    zs_scaled = scaler.fit_transform(zs)
    pca = PCA(n_components=2)
    zs_pca = pca.fit_transform(zs_scaled)

    fig3, ax3 = plt.subplots()
    colors = {"sine":"C0", "linear":"C1", "quadratic":"C2", "fib":"C3", "noise":"C4"}
    for k in set(meta):
        mask = [m==k for m in meta]
        ax3.scatter(zs_pca[mask,0], zs_pca[mask,1], label=k, alpha=0.7)
    ax3.set_title("PCA of latent vectors (z) — clusters by generative family")
    ax3.legend()
    st.pyplot(fig3)

    try:
        st.markdown("**t-SNE** is another method for visualizing the latent space, often better at revealing complex, non-linear relationships and clusters in the data compared to PCA.")
        tsne = TSNE(n_components=2, perplexity=30, max_iter=600, init="pca", random_state=0)
        subset = zs_scaled if zs_scaled.shape[0] <= 500 else zs_scaled[np.random.choice(zs_scaled.shape[0], 500, replace=False)]
        zs_tsne = tsne.fit_transform(subset)
        meta_subset = meta if subset.shape[0]==zs_scaled.shape[0] else [meta[i] for i in np.random.choice(len(meta), 500, replace=False)]
        fig4, ax4 = plt.subplots()
        for k in set(meta_subset):
            mask = [m==k for m in meta_subset]
            ax4.scatter(zs_tsne[mask,0], zs_tsne[mask,1], label=k, alpha=0.7)
        ax4.set_title("t-SNE (subset) of latents")
        ax4.legend()
        st.pyplot(fig4)
    except Exception as e:
        st.write("t-SNE failed or too slow in this environment:", e)

    with torch.no_grad():
        input_t = torch.tensor(example_seq.reshape(1, -1), dtype=torch.float32).to(device)
        recon_example, z_example = model(input_t)
        recon_example = recon_example.cpu().numpy().reshape(-1)
        z_example = z_example.cpu().numpy().reshape(-1)

    st.markdown("---")
    st.markdown("### Analysis of a Single Example")
    st.markdown("Below is the compressed **latent vector (z)** for the example sequence shown at the very top. This small vector contains all the information the model needs to reconstruct the original sequence.")
    st.write(np.round(z_example, 6).tolist())

    st.markdown("The graph below compares the original example sequence with the version reconstructed from the compressed latent vector. The closer the two lines are, the more accurately the model has learned to compress and decompress the data without losing information.")
    st.markdown("""
> **Note:** This AI-based compression is **lossy** — the reconstruction may not exactly match the original data.  
> In real-world deployments, the residual loss can be stored separately and minimized by using a larger neural model with higher-capacity latents.
""")

    fig5, ax5 = plt.subplots()
    ax5.plot(example_seq, label="original")
    ax5.plot(recon_example, label="reconstruction")
    ax5.set_title("Original vs Reconstruction (example)")
    ax5.legend()
    st.pyplot(fig5)

    st.markdown("---")
    st.markdown("### Measuring Compression")
    st.markdown("To measure a real-world compression ratio, we compare the size of the original data (after standard compression like `zstd`) with the size of our model's compressed latent vector (also after standard compression). A higher ratio means our model provides better compression.")
    def quantize_z(z, bits=8):
        zmin, zmax = z.min(), z.max()
        if zmax == zmin:
            return np.zeros_like(z, dtype=np.uint8), (zmin, zmax)
        q = np.round((z - zmin) / (zmax - zmin) * (2**bits - 1)).astype(np.uint8)
        return q, (zmin, zmax)

    zq, zrange = quantize_z(z_example, bits=8)
    raw_bytes = example_seq.astype(np.float32).tobytes()
    q_bytes = zq.tobytes()
    header = json.dumps({"dtype":"float32","length":len(example_seq),"zrange": (float(zrange[0]), float(zrange[1])) }).encode()
    comp_raw = compress_bytes(raw_bytes)
    comp_q = compress_bytes(header + q_bytes)
    st.write(f"Raw size (bytes): {len(raw_bytes)}; Compressed (zlib/zstd) size: {len(comp_raw)}")
    st.write(f"Quantized latent size (bytes): {len(q_bytes)}; Compressed (zlib/zstd) size: {len(comp_q)}")
    compression_ratio = len(comp_raw) / max(1, len(comp_q))
    st.write(f"**Approx compression ratio (raw_compressed / latent_compressed): {compression_ratio:.2f}x**")
    st.info("This is a pragmatic metric: real algorithmic compression stores program tokens (text) + params; here we quantify the learned latent as a proxy.")

    st.write("---")
    st.write("### Symbolic regression (try to extract an analytic generator)")
    st.markdown("The ultimate goal of algorithmic compression is to find the simplest 'program' that can generate the data. Here, we use symbolic regression to try and reverse-engineer a simple mathematical formula (like `A * sin(B*x + C)`) from the model's reconstructed sequence. If successful, this formula is the 'program'.")
    x_vals = np.linspace(0,1,length)
    best_fit = None
    if HAS_PYSR:
        st.write("PySR detected — running symbolic regression (this may take time)...")
        try:
            pysr_model = PySRRegressor(
                niterations=40,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sin", "cos", "exp", "log"],
                model_selection="best",
                verbosity=0
            )
            pysr_model.fit(x_vals.reshape(-1,1), recon_example)
            expr = pysr_model.sympy()
            st.write("PySR expression (sympy):")
            st.write(expr)
            best_fit = ("pysr", expr)
        except Exception as e:
            st.write("PySR failed:", e)
    else:
        st.write("PySR not available — using curve_fit on common families (sine/linear/quadratic).")
        cf = try_curvefit(x_vals, recon_example)
        if cf is not None:
            kind_fit, params_fit, mse = cf
            st.write(f"Best curve-fit: {kind_fit} with params {np.round(params_fit,6)} (MSE={mse:.6e})")
            best_fit = (kind_fit, params_fit)
        else:
            st.write("No simple parametric fit found with curve_fit.")

    if best_fit is not None:
        if best_fit[0] == "pysr":
            pass
        else:
            st.markdown("This final graph compares the model's reconstructed sequence to the mathematical formula found by the curve-fitting process. A close match suggests that our model has successfully captured the underlying algorithmic nature of the data.")
            kind_fit, params_fit = best_fit
            if kind_fit == "sine":
                fit_y = synth_sine(x_vals, *params_fit)
            elif kind_fit == "linear":
                fit_y = synth_linear(x_vals, *params_fit)
            elif kind_fit == "quadratic":
                fit_y = synth_quadratic(x_vals, *params_fit)
            else:
                fit_y = None
            if fit_y is not None:
                fig6, ax6 = plt.subplots()
                ax6.plot(recon_example, label="reconstruction")
                ax6.plot(fit_y, label=f"fitted ({kind_fit})")
                ax6.set_title("Reconstruction vs Fitted analytic function")
                ax6.legend()
                st.pyplot(fig6)

    st.session_state['model'] = model
    st.session_state['scaler'] = scaler
    st.session_state['pca'] = pca
    st.session_state['zs'] = zs
    st.session_state['meta'] = meta

if save_model_btn:
    if 'model' in st.session_state:
        torch.save(st.session_state['model'].state_dict(), MODEL_PATH)
        st.success(f"Saved model weights to {MODEL_PATH}")
    else:
        st.warning("No trained model in session. Train first.")

if load_model_btn:
    if os.path.exists(MODEL_PATH):
        try:
            model = Compressor(length=length, latent_dim=latent_dim)
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            model.eval()
            st.session_state['model'] = model
            st.success("Model loaded into session.")
        except Exception as e:
            st.error("Failed to load model: " + str(e))
    else:
        st.warning("No saved model file found.")
