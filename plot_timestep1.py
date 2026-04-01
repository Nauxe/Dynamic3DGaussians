import matplotlib.pyplot as plt
import numpy as np
import os

log_dir = "./output/sphere-bounce-5/sphere-bounce-5"
t = 1

log_file = os.path.join(log_dir, f"timestep_{t}_loss.txt")
data = np.genfromtxt(log_file, delimiter=',', skip_header=1)

if data.size == 0:
    print("No data found")
    exit()

mask = data[:, 0] <= 600
iterations = data[mask, 0]

loss_cols = ['Im', 'Seg', 'Rigid', 'Rot', 'Iso', 'Floor']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f"Loss Components - Timestep {t} (Iterations 0-600)", fontsize=14)

losses = {
    'Im': data[mask, 2],
    'Seg': data[mask, 3],
    'Rigid': data[mask, 4],
    'Rot': data[mask, 5],
    'Iso': data[mask, 6],
    'Floor': data[mask, 7],
}

for idx, (col, color) in enumerate(zip(loss_cols, colors)):
    ax = axes[idx // 3, idx % 3]
    ax.plot(iterations, losses[col], color=color, linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(f'{col} Loss')
    ax.set_title(f'{col} Loss')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(log_dir, "timestep1_losses_iter0_600.png"), dpi=150)
plt.close()

fig2, ax2 = plt.subplots(figsize=(10, 6))
for col, color in zip(loss_cols, colors):
    ax2.plot(iterations, losses[col], label=col, linewidth=1.5)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss Value')
ax2.set_title(f'All Loss Components - Timestep {t} (Iterations 0-600)')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "timestep1_all_losses_iter0_600.png"), dpi=150)
plt.close()

fig3, ax3 = plt.subplots(figsize=(10, 6))
psnr = data[mask, 10]
ax3.plot(iterations, psnr, color='#2ca02c', linewidth=1.5)
ax3.set_xlabel('Iteration')
ax3.set_ylabel('PSNR (dB)')
ax3.set_title(f'PSNR - Timestep {t} (Iterations 0-600)')
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "timestep1_psnr_iter0_600.png"), dpi=150)
plt.close()

print("Generated:")
print("  - timestep1_losses_iter0_600.png")
print("  - timestep1_all_losses_iter0_600.png")
print("  - timestep1_psnr_iter0_600.png")
