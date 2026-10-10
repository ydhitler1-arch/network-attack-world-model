"""Draw docs/architecture.png (needs matplotlib).   python docs/make_architecture_figure.py"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(7.4, 3.2), dpi=220)
ax.set_xlim(0, 100)
ax.set_ylim(0, 44)
ax.axis("off")


def box(x, y, w, h, title, sub="", fc="#eef3fa", ec="#34506f", fs=7.2, sub_dy=5.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="top", fontsize=fs, fontweight="bold", color="#1b2a3a")
    if sub:
        ax.text(x + w / 2, y + h - sub_dy, sub, ha="center", va="top", fontsize=fs - 1.3, color="#2b3b4c", linespacing=1.25)


def arrow(x0, y0, x1, y1, lw=0.9, color="#34506f"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=lw, color=color))


box(1, 26, 16, 16, "Input", "flow CSV\n(CIC-IDS2017)\nor PCAP\n(Scapy to flows)")
box(21, 26, 18, 16, "State S_t", "60 s windows,\n43 features: flow stats,\nscan signatures, TCP-\nwindow, packet-size", fc="#f4efe4")
box(43, 10, 33, 32, "World model (LSTM, 128 hidden)", "reads S_(t-5) .. S_t, then rolls out K = 6 steps", fc="#e8f3ec", fs=6.8)
box(45, 12.5, 14.5, 12, "dynamics head", "N(mu, sigma)\nover next state", fc="#ffffff", fs=6.1, sub_dy=4.6)
box(60.5, 12.5, 14.5, 12, "readout heads", "P(attack) and\nATT&CK stage", fc="#ffffff", fs=6.1, sub_dy=4.6)
arrow(52.2, 24.9, 52.2, 33.2, lw=0.8, color="#7a8a7f")
ax.text(53.2, 29.2, "predicted mean state is fed\nback in (steps k = 1..6)", fontsize=5.3, color="#4d5d52", va="center", linespacing=1.2)
box(80, 26, 19, 16, "Prediction", "P(infiltration) for\nt, t+1 ... t+6 and\nATT&CK stage\nper window")
box(80, 3, 19, 16, "Explanation", "SHAP (expected\ngradients) per feature\nand per lag, for every\nprediction", fc="#fbeaea")
box(21, 3, 18, 14, "Offline demo", "Streamlit: timeline,\nflagged windows,\nflagged hosts and flows", fc="#f1ecf8")
arrow(17.3, 34, 20.7, 34)
arrow(39.3, 34, 42.7, 34)
arrow(76.3, 34, 79.7, 34)
arrow(89.5, 25.7, 89.5, 19.3)
arrow(79.7, 6.5, 39.4, 6.5)
ax.text(59, 4.3, "predictions and explanations", ha="center", fontsize=5.4, color="#555")
ax.text(1, 21.5, "Trained on CIC-IDS2017\n(5 capture days, 2,453\nwindows, 264 attack\nwindows). Benchmarked vs\nlogistic regression and an\nLSTM classifier.", fontsize=5.5, color="#444",
        va="top", linespacing=1.3)
fig.savefig("docs/architecture.png", bbox_inches="tight", pad_inches=0.04)
print("saved docs/architecture.png")
