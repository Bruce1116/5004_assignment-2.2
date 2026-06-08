import matplotlib.pyplot as plt


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Value", save_path=None):
    plt.figure()
    plt.plot(range(1, len(metric_values) + 1), metric_values, marker="o")
    plt.title(title)
    plt.xlabel("Chunk")
    plt.ylabel(ylabel)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"), save_path=None):
    plt.figure()
    plt.plot(range(1, len(metric1) + 1), metric1, marker="o", label=labels[0])
    plt.plot(range(1, len(metric2) + 1), metric2, marker="o", label=labels[1])
    plt.xlabel("Chunk")
    plt.ylabel("Metric")
    plt.title("Model comparison")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def plot_predictions_vs_ground_truth(y_true, y_pred, save_path=None):
    plt.figure()
    plt.plot(y_true, label="Ground truth", marker="o")
    plt.plot(y_pred, label="Prediction", marker="x")
    plt.xlabel("Sample")
    plt.ylabel("Class")
    plt.title("Predictions vs ground truth")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

