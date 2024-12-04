import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class MetricAnalysis:
    """
    A class to analyze and visualize training metrics and classification statistics for RNN models.
    """

    @staticmethod
    def plot_metrics(history, output_dir="metrics_plots", prefix="model"):
        """
        Plot accuracy and loss metrics from the training history.

        Args:
            history (dict): A dictionary containing metric values (returned from `model.fit().history`).
            output_dir (str): Directory where plots will be saved. Defaults to "metrics_plots".
            prefix (str): A prefix for plot filenames. Defaults to "model".
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Plot accuracy
        if "accuracy" in history and "val_accuracy" in history:
            plt.figure()
            plt.plot(history["accuracy"], label="Training Accuracy")
            plt.plot(history["val_accuracy"], label="Validation Accuracy")
            plt.title("Model Accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(output_dir, f"{prefix}_accuracy.png"))
            plt.close()

        # Plot loss
        if "loss" in history and "val_loss" in history:
            plt.figure()
            plt.plot(history["loss"], label="Training Loss")
            plt.plot(history["val_loss"], label="Validation Loss")
            plt.title("Model Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(output_dir, f"{prefix}_loss.png"))
            plt.close()

    @staticmethod
    def summarize_metrics(history, fold_no=None):
        """
        Print a summary of the metrics at the last epoch.

        Args:
            history (dict): A dictionary containing metric values (returned from `model.fit().history`).
            fold_no (int, optional): Fold number for cross-validation. Defaults to None.
        """
        last_epoch = len(history["accuracy"])
        summary = f"Metrics Summary for {'Fold ' + str(fold_no) if fold_no else 'Model'}:\n"
        summary += f"- Final Training Accuracy: {history['accuracy'][-1]:.4f}\n"
        summary += f"- Final Validation Accuracy: {history['val_accuracy'][-1]:.4f}\n"
        summary += f"- Final Training Loss: {history['loss'][-1]:.4f}\n"
        summary += f"- Final Validation Loss: {history['val_loss'][-1]:.4f}\n"
        print(summary)

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, output_dir="metrics_plots", prefix="model"):
        """
        Generate and save a confusion matrix plot.

        Args:
            y_true (np.array): Ground truth labels.
            y_pred (np.array): Predicted labels.
            output_dir (str): Directory where the confusion matrix plot will be saved.
            prefix (str): A prefix for the plot filename.
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.savefig(os.path.join(output_dir, f"{prefix}_confusion_matrix.png"))
        plt.close()

    @staticmethod
    def classification_report_summary(y_true, y_pred):
        """
        Print and return a classification report.

        Args:
            y_true (np.array): Ground truth labels.
            y_pred (np.array): Predicted labels.

        Returns:
            dict: Classification report as a dictionary.
        """
        report = classification_report(y_true, y_pred, output_dict=True)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        return report
