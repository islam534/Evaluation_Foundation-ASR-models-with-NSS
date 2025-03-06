import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Data from Table 1 (means)
categories = ['WER (Inverted)', 'CER (Inverted)', 'Exact Match', 'Seq. Similarity', 'Jaccard Sim.']
models = {
    'Conformer': [0.42, 0.69, 0.11, 0.70, 0.34],  # Normalized means
    'Hubert': [0.47, 0.72, 0.30, 0.85, 0.43],
    'wav2vec2': [0.13, 0.11, 0.00, 0.00, 0.00],
    'Whisper': [0.14, 0.00, 1.00, 1.00, 1.00]
}
sds = {  # Approximate SDs (adjust with actual data)
    'Conformer': [0.1, 0.05, 0.05, 0.05, 0.05],
    'Hubert': [0.09, 0.04, 0.06, 0.04, 0.04],
    'wav2vec2': [0.12, 0.15, 0.03, 0.06, 0.03],
    'Whisper': [0.3, 0.4, 0.1, 0.05, 0.05]  # Larger SDs for variability
}

# Number of axes
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # Close the plot

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw axes and labels
plt.xticks(angles[:-1], categories)
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
plt.ylim(0, 1)

# Plot each model
colors = ['b', 'g', 'r', 'purple']
for idx, (model, values) in enumerate(models.items()):
    values += values[:1]  # Close the plot
    sd_values = sds[model] + sds[model][:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[idx])
    ax.fill_between(angles, [v - sd for v, sd in zip(values, sd_values)], 
                    [v + sd for v, sd in zip(values, sd_values)], color=colors[idx], alpha=0.2)

# Add legend and title
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.title("Performance Profiles of ASR Models with Variability", size=14, pad=20)
plt.tight_layout()
plt.show()