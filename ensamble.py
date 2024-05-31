import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from scipy.stats import mode
import os


def load_saved_outputs(model_filename):
    outputs_list = np.load(model_filename)
    return torch.tensor(outputs_list)


def normalize_outputs(outputs):
    softmax = torch.nn.Softmax(dim=1)
    return softmax(outputs)


def calculate_metrics(outputs, labels):
    _, predicted_labels = torch.max(outputs, 1)
    predicted_labels = predicted_labels.numpy()
    f1 = f1_score(labels, predicted_labels, average="weighted")
    balanced_acc = balanced_accuracy_score(labels, predicted_labels)
    acc = accuracy_score(labels, predicted_labels)
    return f1, balanced_acc, acc, predicted_labels


def ensemble_outputs_mean(output_list):
    normalized_output_list = [normalize_outputs(output) for output in output_list]
    total_outputs = torch.stack(normalized_output_list).mean(dim=0)
    return total_outputs


def ensemble_outputs_vote(output_list):
    normalized_output_list = [normalize_outputs(output) for output in output_list]
    predicted_labels_list = [
        torch.argmax(output, dim=1) for output in normalized_output_list
    ]
    predicted_labels_stack = torch.stack(predicted_labels_list, dim=1)
    voted_predictions, _ = mode(predicted_labels_stack.numpy(), axis=1)
    voted_predictions = torch.tensor(voted_predictions.squeeze(), dtype=torch.long)
    return voted_predictions


def generate_confusion_matrix(actual, predicted, class_names, model_name, acc, mode):
    cm = confusion_matrix(actual, predicted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    num_classes = len(class_names)
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    fig = ax.get_figure()
    fig.patch.set_facecolor("none")
    ax.patch.set_facecolor("none")
    disp.plot(ax=ax, cmap="viridis", xticks_rotation="vertical")
    plt.xticks(ticks=np.arange(num_classes), labels=class_names, rotation=90)
    plt.yticks(ticks=np.arange(num_classes), labels=class_names)
    plt.tight_layout()
    output_dir = f"plots/{mode}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        f"{output_dir}/confusion_matrix_{model_name}_{acc:.4f}.png",
        transparent=True,
    )


def plot_metrics(x, y, metric_name, mode):
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("Number of Models in Ensemble")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs Number of Models in Ensemble ({mode})")
    plt.tight_layout()
    output_dir = f"plots/{mode}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{metric_name.lower().replace(' ', '_')}_{mode}.png")


def main(mode="mean"):
    actual_labels = np.load("vit/outputs/actual_labels.npy")
    unique_classes = np.unique(actual_labels)
    class_names = [f"Class{i}" for i in range(len(unique_classes))]

    model_filenames = [
        f"vit/outputs/outputs_list_best_model_loss_vit_0_acc_0_train_all_True_{i}.npy"
        for i in range(2, 12)
    ]
    # cnn/outputs/outputs_list_best_model_loss_cnn_loss1.npy
    # resnet/outputs/outputs_list_best_model_loss_resnet_loss1.npy
    model_filenames.append("cnn/outputs/outputs_list_best_model_loss_cnn_loss1.npy")
    model_filenames.append(
        "resnet/outputs/outputs_list_best_model_loss_resnet_loss1.npy"
    )
    all_outputs = [load_saved_outputs(filename) for filename in model_filenames]

    metrics_filename = f"metrics_{mode}.txt"
    with open(metrics_filename, "w") as f:
        # Initialize lists to store metrics for plotting
        num_models = list(range(1, len(all_outputs) + 1))
        f1_scores = []
        balanced_accuracies = []
        accuracies = []

        # Calculate metrics for individual models
        for i, outputs in enumerate(all_outputs):
            normalized_outputs = normalize_outputs(outputs)
            f1, balanced_acc, acc, predicted_labels = calculate_metrics(
                normalized_outputs, actual_labels
            )
            f1_scores.append(f1)
            balanced_accuracies.append(balanced_acc)
            accuracies.append(acc)
            f.write(
                f"Model {i+1} - F1: {f1:.4f}, Balanced Accuracy: {balanced_acc:.4f}, Accuracy: {acc:.4f}\n"
            )
            print(
                f"Model {i+1} - F1: {f1:.4f}, Balanced Accuracy: {balanced_acc:.4f}, Accuracy: {acc:.4f}"
            )
            generate_confusion_matrix(
                actual_labels, predicted_labels, class_names, f"model_{i+1}", acc, mode
            )

        # Calculate metrics for ensemble models
        for i in range(2, len(all_outputs) + 1):
            if mode == "mean":
                ensemble_outputs_list = ensemble_outputs_mean(all_outputs[:i])
                f1, balanced_acc, acc, predicted_labels = calculate_metrics(
                    ensemble_outputs_list, actual_labels
                )
            elif mode == "vote":
                voted_predictions = ensemble_outputs_vote(all_outputs[:i])
                f1 = f1_score(actual_labels, voted_predictions, average="weighted")
                balanced_acc = balanced_accuracy_score(actual_labels, voted_predictions)
                acc = accuracy_score(actual_labels, voted_predictions)
                predicted_labels = voted_predictions.numpy()
            f1_scores.append(f1)
            balanced_accuracies.append(balanced_acc)
            accuracies.append(acc)
            f.write(
                f"Ensemble of first {i} models - F1: {f1:.4f}, Balanced Accuracy: {balanced_acc:.4f}, Accuracy: {acc:.4f}\n"
            )
            print(
                f"Ensemble of first {i} models - F1: {f1:.4f}, Balanced Accuracy: {balanced_acc:.4f}, Accuracy: {acc:.4f}"
            )
            generate_confusion_matrix(
                actual_labels, predicted_labels, class_names, f"ensemble_{i}", acc, mode
            )

        # Update num_models to include ensemble points
        num_models = list(range(1, len(all_outputs) + 1))

        # Plot metrics
        plot_metrics(num_models, f1_scores[len(all_outputs) - 1 :], "F1 Score", mode)
        plot_metrics(
            num_models,
            balanced_accuracies[len(all_outputs) - 1 :],
            "Balanced Accuracy",
            mode,
        )
        plot_metrics(num_models, accuracies[len(all_outputs) - 1 :], "Accuracy", mode)


if __name__ == "__main__":
    main(mode="mean")  # Change to "mean" or "vote" as needed
