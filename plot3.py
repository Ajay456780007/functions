import numpy as np
import matplotlib.pyplot as plt
A=np.load("../Analysis/ROC_Analysis/Concated_epochs/Zea_mays/metrics_epochs_100.npy")
print(A)
bar_width=0.1
model=["KNN","CNN","RESNET","RNN","NN","CNN","GOP","UJP"]
colors=["#c45161", "#e094a0", "#f2b6c0", "#f2dde1", "#cbc7d8", "#8db7d2", "#5e62a9", "#434279"]
percentage=[40,50,60,70,80,90]
per_len=A.shape[1]
percentages=percentage[:per_len]
model_len=A.shape[0]
models=model[:model_len]
x=np.arange(len(percentages))
plt.figure(figsize=(14,8))
for i in range(A.shape[0]):
    plt.bar(x+i*bar_width,A[i],width=bar_width,label=model[i],color=colors[i])
center_shift = (A.shape[0] * bar_width) / 2 - bar_width / 2
plt.xticks(x + center_shift, percentages)
plt.xlabel("Training Perecentage(%)")
plt.ylabel("Accuracy(%)")
plt.title("Model Accuracy vs Training Data Percentage (Bar Graph)")
plt.legend(loc="upper left")
plt.grid(True,linestyle='--',alpha=0.1)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,8))
for j in range(A.shape[0]):
    plt.plot(percentages,A[j],marker='o',label=model[j],color=colors[j])

plt.xlabel("Training Data Percentage")
plt.ylabel("Accuracy")
plt.title("Model Accuracy vs Training Data Percentage (Line Graph)")
plt.legend(loc='upper left')
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
