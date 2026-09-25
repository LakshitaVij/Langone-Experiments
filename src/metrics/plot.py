import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from torch.nn import Softmax


def plot_roc_curve(
    fpr, tpr, thresholds, opt_threshold, balanced_acc, roc_auc, save_path=None
):
    """Plot the ROC curve for a binary classification task.

    args:
        fpr: list of false positive rates
        tpr: list of true positive rates
        thresholds: threshold values
        opt_threshold: optimal threshold (wrt balanced accuracy)
        balanced_acc: balanced accuracy at optimal threshold
        roc_auc: area under the ROC curve
        save_path: file path to save the plot
    """
    # get index of optimal threshold
    idx = np.argwhere(thresholds == opt_threshold)[0][0]

    # set up figure
    plt.figure()

    # plot the ROC curve
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")

    # plot the optimal threshold
    plt.plot(fpr[idx], tpr[idx], "ro", label=f"Threshold = {opt_threshold:.2f}")

    # plot the 45 degree line
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)

    # add labels
    plt.grid()
    plt.xlabel(f"False Positive Rate: ({fpr[idx]:.2f}@threshold)")
    plt.ylabel(f"True Positive Rate: (tpr={tpr[idx]:.2f}@threshold)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig(save_path)


def plot_precision_recall_curve(precision, recall, thresholds, save_path=None):
    """Plot the precision-recall curve for a binary classification task.

    args:
        precision: list of precision values
        recall: list of recall values
        thresholds: threshold values
        save_path: file path to save the plot
    """
    # Plot the precision-recall curve
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(save_path)


def plot_pirads_cm(
    preds: np.ndarray,
    targets: np.ndarray,
    pirads: np.ndarray,
    epoch_count,
    save_path=None,
):
    """Plot the PIRADS breakdown of TP, FP, TN, FN.

    args:
        preds: list of torch tensors of shape (batch_size)
        targets: list targets of shape (batch_size)
        pirads: list of PIRADS scores of shape (batch_size)
        epoch_count: int, current epoch
        save_path: file path to save the plot
    """
    # get indices of TP, FP, TN, FN
    tp_idx = np.argwhere((preds == 1) & (targets == 1)).flatten()
    fp_idx = np.argwhere((preds == 1) & (targets == 0)).flatten()
    tn_idx = np.argwhere((preds == 0) & (targets == 0)).flatten()
    fn_idx = np.argwhere((preds == 0) & (targets == 1)).flatten()

    # get PIRADS scores for each
    if tp_idx.size == 0:
        tp_pirads = []
    else:
        tp_pirads = pirads[tp_idx]
    if fp_idx.size == 0:
        fp_pirads = []
    else:
        fp_pirads = pirads[fp_idx]
    if tn_idx.size == 0:
        tn_pirads = []
    else:
        tn_pirads = pirads[tn_idx]
    if fn_idx.size == 0:
        fn_pirads = []
    else:
        fn_pirads = pirads[fn_idx]

    # set up figure
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # plot TP
    ax[0, 0].hist(tp_pirads, bins=5, range=(1, 5))
    ax[0, 0].set_title(f"TP: {len(tp_pirads)} samples")
    # plot FP
    ax[0, 1].hist(fp_pirads, bins=5, range=(1, 5))
    ax[0, 1].set_title(f"FP: {len(fp_pirads)} samples")
    # plot TN
    ax[1, 0].hist(tn_pirads, bins=5, range=(1, 5))
    ax[1, 0].set_title(f"TN: {len(tn_pirads)} samples")
    # plot FN
    ax[1, 1].hist(fn_pirads, bins=5, range=(1, 5))
    ax[1, 1].set_title(f"FN: {len(fn_pirads)} samples")

    # add title
    fig.suptitle(f"PIRADS Breakdown: Epoch {epoch_count}")

    # save figure
    plt.savefig(save_path)


def save_preds(preds, targets, max_pirads, acc_nums, save_dir, epoch_count, save_name=None):
    """Save predictions to a file.

    args:
        preds: list of torch tensors of shape (batch_size, 2)
        targets: list targets of shape (batch_size)
        save_dir: file path to save the predictions
        epoch_count: int, current epoch
        save_name: str, name of the file to save the predictions
    """
    # use softmax to get class probabilities
    softmax = Softmax(dim=1)
    pred_probs = softmax(preds)

    # save predictions and targets to a csv
    results = DataFrame(
        {
            "Predictions": pred_probs.cpu().tolist(),
            "Targets": targets.cpu().tolist(),
            "maxPIRADS": max_pirads.cpu().tolist(),
            "AccessionNumber": acc_nums.cpu().tolist()
        }
    )
    if save_name is not None:
        save_path = save_dir + f"/{save_name}_epoch_{epoch_count}.csv"
    else:
        save_path = save_dir + f"/preds_epoch_{epoch_count}.csv"
    results.to_csv(save_path, index=False)
