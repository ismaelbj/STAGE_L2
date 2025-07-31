import csv
import sys
import os
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python plot_metrics.py <path_to_csv>")
    sys.exit(1)

csv_file = sys.argv[1]

# === Nom du modèle ===
if "dmpnn" in csv_file:
    model_name = "dmpnn"
elif "gat" in csv_file:
    model_name = "gat"
elif "gin" in csv_file:
    model_name = "gin"
elif "transformer" in csv_file:
    model_name = "transformer"
elif "graphsage" in csv_file:
    model_name = "graphsage"
else:
    model_name = "unknown_model"

output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)
output_file = f"training_curves_{model_name}_optimized.png"


epochs = []
train_loss, val_loss, test_loss = [], [], []
train_acc, val_acc, test_acc = [], [], []
train_f1, val_f1, test_f1 = [], [], []
val_auroc, test_auroc = [], []
val_aupr, test_aupr = [], []
val_prec, test_prec = [], []
val_rec, test_rec = [], []

with open(csv_file, mode="r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        train_loss.append(float(row["train_loss"]))
        val_loss.append(float(row["val_loss"]))
        test_loss.append(float(row["test_loss"]))
        train_acc.append(float(row["train_acc"]))
        val_acc.append(float(row["val_acc"]))
        test_acc.append(float(row["test_acc"]))
        train_f1.append(float(row.get("train_f1_score", 0)))
        val_f1.append(float(row.get("val_f1_score", 0)))
        test_f1.append(float(row.get("test_f1_score", 0)))
        val_auroc.append(float(row.get("val_auroc", 0)))
        test_auroc.append(float(row.get("test_auroc", 0)))
        val_aupr.append(float(row.get("val_aupr", 0)))
        test_aupr.append(float(row.get("test_aupr", 0)))
        val_prec.append(float(row.get("val_prec", 0)))
        test_prec.append(float(row.get("test_prec", 0)))
        val_rec.append(float(row.get("val_rec", 0)))
        test_rec.append(float(row.get("test_rec", 0)))


plt.figure(figsize=(21, 10)) 

#Loss
plt.subplot(2, 4, 1)
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Val Loss")
plt.plot(epochs, test_loss, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.legend()

#Accuracy
plt.subplot(2, 4, 2)
plt.plot(epochs, train_acc, label="Train Acc")
plt.plot(epochs, val_acc, label="Val Acc")
plt.plot(epochs, test_acc, label="Test Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curves")
plt.legend()

#AUROC
plt.subplot(2, 4, 3)
plt.plot(epochs, val_auroc, label="Val AUROC")
plt.plot(epochs, test_auroc, label="Test AUROC")
plt.xlabel("Epoch")
plt.ylabel("AUROC")
plt.title("AUROC")
plt.legend()

#AUPR
plt.subplot(2, 4, 4)
plt.plot(epochs, val_aupr, label="Val AUPR")
plt.plot(epochs, test_aupr, label="Test AUPR")
plt.xlabel("Epoch")
plt.ylabel("AUPR")
plt.title("AUPR")
plt.legend()

#Precision
plt.subplot(2, 4, 5)
plt.plot(epochs, val_prec, label="Val Precision")
plt.plot(epochs, test_prec, label="Test Precision")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.title("Precision")
plt.legend()

#Recall
plt.subplot(2, 4, 6)
plt.plot(epochs, val_rec, label="Val Recall")
plt.plot(epochs, test_rec, label="Test Recall")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.title("Recall")
plt.legend()

#F1-Score
plt.subplot(2, 4, 7)
plt.plot(epochs, val_f1, label="Val F1")
plt.plot(epochs, test_f1, label="Test F1")
plt.xlabel("Epoch")
plt.ylabel("F1-Score")
plt.title("F1-Score")
plt.legend()

plt.tight_layout()

save_path = os.path.join(output_dir, output_file)
plt.savefig(save_path)
print(f"Courbes sauvegardées ici : {save_path}")