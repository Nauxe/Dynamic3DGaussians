import matplotlib.pyplot as plt
import numpy as np
import os

log_dir = "./output/sphere-bounce-5/sphere-bounce-5"

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Loss Component Analysis - Sphere Bounce", fontsize=14)

loss_cols = ['Im', 'Seg', 'Rigid', 'Rot', 'Iso', 'Floor']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for idx, (col, color) in enumerate(zip(loss_cols, colors)):
    ax = axes[idx // 3, idx % 3]
    
    for t in range(5):
        log_file = os.path.join(log_dir, f"timestep_{t}_loss.txt")
        data = np.genfromtxt(log_file, delimiter=',', skip_header=1)
        
        if data.size == 0:
            continue
        
        iterations = data[:, 0]
        values = data[:, 2 + idx]
        
        ax.plot(iterations, values, label=f'Timestep {t}', alpha=0.7)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel(f'{col} Loss')
    ax.set_title(f'{col} Loss over Timesteps')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(log_dir, "loss_components_analysis.png"), dpi=150)
plt.close()

fig2, ax2 = plt.subplots(figsize=(12, 6))
for t in range(5):
    log_file = os.path.join(log_dir, f"timestep_{t}_loss.txt")
    data = np.genfromtxt(log_file, delimiter=',', skip_header=1)
    
    if data.size == 0:
        continue
    
    iterations = data[:, 0]
    total_loss = data[:, 1]
    
    ax2.plot(iterations, total_loss, label=f'Timestep {t}', alpha=0.7)

ax2.set_xlabel('Iteration')
ax2.set_ylabel('Total Loss')
ax2.set_title('Total Loss over Timesteps')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "total_loss_comparison.png"), dpi=150)
plt.close()

fig_psnr, ax_psnr = plt.subplots(figsize=(12, 6))
for t in range(5):
    log_file = os.path.join(log_dir, f"timestep_{t}_loss.txt")
    data = np.genfromtxt(log_file, delimiter=',', skip_header=1)
    
    if data.size == 0:
        continue
    
    iterations = data[:, 0]
    psnr = data[:, 10]
    
    ax_psnr.plot(iterations, psnr, label=f'Timestep {t}', alpha=0.7)

ax_psnr.set_xlabel('Iteration')
ax_psnr.set_ylabel('PSNR (dB)')
ax_psnr.set_title('PSNR over Timesteps')
ax_psnr.legend()
ax_psnr.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "psnr_comparison.png"), dpi=150)
plt.close()

for t in range(5):
    log_file = os.path.join(log_dir, f"timestep_{t}_loss.txt")
    data = np.genfromtxt(log_file, delimiter=',', skip_header=1)
    
    if data.size == 0:
        continue
    
    fig_psnr_t, ax_psnr_t = plt.subplots(figsize=(10, 5))
    iterations = data[:, 0]
    psnr = data[:, 10]
    
    ax_psnr_t.plot(iterations, psnr, color='#2ca02c', linewidth=1.5)
    ax_psnr_t.set_xlabel('Iteration')
    ax_psnr_t.set_ylabel('PSNR (dB)')
    ax_psnr_t.set_title(f'PSNR - Timestep {t}')
    ax_psnr_t.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"psnr_timestep_{t}.png"), dpi=150)
    plt.close()

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

ax = axes3[0]
t = 1
log_file = os.path.join(log_dir, f"timestep_{t}_loss.txt")
data = np.genfromtxt(log_file, delimiter=',', skip_header=1)

if data.size > 0:
    iterations = data[:, 0]
    losses = {
        'Im': data[:, 2],
        'Seg': data[:, 3],
        'Rigid': data[:, 4],
        'Rot': data[:, 5],
        'Iso': data[:, 6],
        'Floor': data[:, 7],
    }
    
    for col, color in zip(loss_cols, colors):
        ax.plot(iterations, losses[col], label=col, alpha=0.8)

ax.set_xlabel('Iteration')
ax.set_ylabel('Loss Value')
ax.set_title(f'All Loss Components - Timestep {t}')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes3[1]
t = 2
log_file = os.path.join(log_dir, f"timestep_{t}_loss.txt")
data = np.genfromtxt(log_file, delimiter=',', skip_header=1)

if data.size > 0:
    iterations = data[:, 0]
    losses = {
        'Im': data[:, 2],
        'Seg': data[:, 3],
        'Rigid': data[:, 4],
        'Rot': data[:, 5],
        'Iso': data[:, 6],
        'Floor': data[:, 7],
    }
    
    for col, color in zip(loss_cols, colors):
        ax.plot(iterations, losses[col], label=col, alpha=0.8)

ax.set_xlabel('Iteration')
ax.set_ylabel('Loss Value')
ax.set_title(f'All Loss Components - Timestep {t}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(log_dir, "all_components_comparison.png"), dpi=150)
plt.close()

weights = {'Im': 1.0, 'Seg': 3.0, 'Rigid': 4.0, 'Rot': 4.0, 'Iso': 2.0, 'Floor': 2.0}

for t in range(5):
    log_file = os.path.join(log_dir, f"timestep_{t}_loss.txt")
    data = np.genfromtxt(log_file, delimiter=',', skip_header=1)
    
    if data.size == 0:
        continue
    
    fig4, axes4 = plt.subplots(2, 3, figsize=(15, 8))
    fig4.suptitle(f"Weighted Loss Contribution - Timestep {t}", fontsize=14)
    
    iterations = data[:, 0]
    
    for idx, (col, color) in enumerate(zip(loss_cols, colors)):
        ax = axes4[idx // 3, idx % 3]
        weighted_loss = data[:, 2 + idx] * weights[col]
        ax.fill_between(iterations, 0, weighted_loss, alpha=0.3, color=color)
        ax.plot(iterations, weighted_loss, color=color, linewidth=1.5)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Weighted Loss')
        ax.set_title(f'{col} × {weights[col]}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"weighted_loss_contribution_timestep_{t}.png"), dpi=150)
    plt.close()

print("Generated:")
print("  - loss_components_analysis.png")
print("  - total_loss_comparison.png")
print("  - psnr_comparison.png")
print("  - all_components_comparison.png")
print("  - weighted_loss_contribution_timestep_0.png")
print("  - weighted_loss_contribution_timestep_1.png")
print("  - weighted_loss_contribution_timestep_2.png")
print("  - weighted_loss_contribution_timestep_3.png")
print("  - weighted_loss_contribution_timestep_4.png")
print("  - psnr_timestep_0.png")
print("  - psnr_timestep_1.png")
print("  - psnr_timestep_2.png")
print("  - psnr_timestep_3.png")
print("  - psnr_timestep_4.png")
