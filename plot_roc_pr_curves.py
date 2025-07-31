import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

models = [
    {"name": "D-MPNN", "y_true": "y_true_dmpnn.npy", "y_scores": "y_scores_dmpnn.npy"},
    {"name": "GIN", "y_true": "y_true_gin.npy", "y_scores": "y_scores_gin.npy"},
    {"name": "Transformer", "y_true": "y_true_transformer.npy", "y_scores": "y_scores_transformer.npy"},
    {"name": "GraphSAGE", "y_true": "y_true_graphsage.npy", "y_scores": "y_scores_graphsage.npy"},
    {"name": "GAT", "y_true": "y_true_gat.npy", "y_scores": "y_scores_gat.npy"} 
]

plt.figure(figsize=(12, 5))

# ROC
plt.subplot(1, 2, 1)
for model in models:
    y_true = np.load(model["y_true"])
    y_scores = np.load(model["y_scores"])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{model['name']} (AUC = {roc_auc:.4f})")

plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")

# PR
plt.subplot(1, 2, 2)
for model in models:
    y_true = np.load(model["y_true"])
    y_scores = np.load(model["y_scores"])
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)
    plt.plot(recall, precision, lw=2, label=f"{model['name']} (AUPR = {aupr:.4f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")

plt.tight_layout()
plt.savefig("roc_pr_multi_models_optimized_1.png")
print("ROC + PR multi-modèles sauvegardés dans roc_pr_multi_models_optimized_1.png")
