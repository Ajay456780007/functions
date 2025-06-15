import matplotlib.pyplot as plt
import numpy as np
A=np.load("../Analysis/Comparative_Analysis/Zea_mays/ACC_1.npy")
print(A)
print(A.shape)
models = [f"Model{i+1}" for i in range(A.shape[0])]
percentages=[40,50,60,70,80,90]

colors = plt.cm.get_cmap('tab10', A.shape[0])  # Different color for each model

# === BAR GRAPH ===
bar_width = 0.1
x = np.arange(len(percentages))

plt.figure(figsize=(14, 6))
for i in range(A.shape[0]):  # loop over models
    plt.bar(x + i*bar_width, A[i], width=bar_width, label=models[i], color=colors(i))

center_shift = (A.shape[0] * bar_width) / 2 - bar_width / 2
plt.xticks(x + center_shift, percentages)
plt.xlabel("Training Percentage")
plt.ylabel("Accuracy")
plt.title("Model Accuracy vs Training Data Percentage (Bar Graph)")
plt.legend(loc='upper left')
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# === LINE GRAPH ===
plt.figure(figsize=(14, 6))
for i in range(A.shape[0]):
    plt.plot(percentages, A[i], marker='o', label=models[i], color=colors(i))

plt.xlabel("Training Data Percentage")
plt.ylabel("Accuracy")
plt.title("Model Accuracy vs Training Data Percentage (Line Graph)")
plt.legend(loc='upper left')
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()