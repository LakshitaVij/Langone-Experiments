"""
This module provides helper functions for sampling slice and volume data.

The functions in this module are used by the `src.data.slice_loader.SliceH5Dataset`
and `src.data.loader.ExamH5Dataset` classes to sample slices and volumes from H5 files.
Exam and slice data are typically imbalanced, so the functions in this module address
score imbalance (PIRADS 1-5) and class imbalance (positive vs. negative).
"""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.sampler import Sampler


class DualLabelBiasedSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.l1 = [
            idx for idx, label in enumerate(dataset.gleason_labels) if label == -1
        ]  # cases with pirads labels only
        self.l2 = [
            idx for idx, label in enumerate(dataset.gleason_labels) if label != -1
        ]  # cases with pirads+gleason labels
        self.batch_size = batch_size
        self.dataset = dataset

    def _resampled_length(self):
        return max(len(self.l1), len(self.l2))

    def __len__(self):
        resampled_length = self._resampled_length()
        batches = 2 * np.ceil(resampled_length / self.batch_size)
        return int(batches)

    def __iter__(self):
        resampled_length = self._resampled_length()

        l1_indices = torch.randint(0, len(self.l1), (resampled_length,))
        resampled_l1 = torch.tensor(self.l1)[l1_indices]

        l2_indices = torch.randint(0, len(self.l2), (resampled_length,))
        resampled_l2 = torch.tensor(self.l2)[l2_indices]

        l1_batches = torch.split(resampled_l1, self.batch_size)
        l2_batches = torch.split(resampled_l2, self.batch_size)

        combined = list(l1_batches + l2_batches)
        batch_indices = torch.randperm(len(combined)).tolist()
        combined = [combined[i].tolist() for i in batch_indices]

        return iter(combined)


def compute_sample_weights(targets, pirads_cutoff):
    """Compute sample weights for a prostate image dataset.

    args:
        targets: ordered list of PIRADS scores for each slice in the dataset
        pirads_cutoff: PIRADS score cutoff
    returns:
        list of sample weights
    """
    num_scores = 5

    # calculate the frequency of each pirads score in the dataset
    score_frequencies = [0, 0, 0, 0, 0]
    for score in targets:
        score_frequencies[score - 1] += 1
    score_frequencies = [freq / len(targets) for freq in score_frequencies]

    # calculate weights based on the frequency of each class and the pirads cutoff
    score_weights = [0, 0, 0, 0, 0]
    score_weights = [1.0 / freq for freq in score_frequencies]

    # calculate a weight to assign to each sample in the dataset
    sample_weights = [score_weights[label - 1] for label in targets]

    # since we are using a binary classification, where ex. PIRADS > 2 is positive,
    # we want to balance the number of positive and negative samples, not strictly
    # the number of slices with a given PIRADS score
    # ex.
    #   if we have a pirads cutoff of 2, the probabilities of the sampler returning
    #   slices with PIRADS scores of
    #       [1, 2, 3, 4, 5]
    #   should be
    #       [.25, .25, .1666, .1666, .1666]
    if pirads_cutoff != 0:
        class_sample_weights = []
        default_frequency = 1 / num_scores
        negative_frequency = 0.5  # desired frequency of negative class

        # calculate the scale factor for each class
        negative_scale = (negative_frequency / pirads_cutoff) / default_frequency
        positive_scale = (
            (1 - negative_frequency) / (num_scores - pirads_cutoff)
        ) / default_frequency
        for target, weight in zip(targets, sample_weights):
            if target <= pirads_cutoff:
                class_sample_weights.append(weight * negative_scale)
            else:
                class_sample_weights.append(weight * positive_scale)
        sample_weights = class_sample_weights

    return sample_weights


def get_sampler(dataset, pirads_cutoff=2):
    """Get a WeightedRandomSampler for the slice dataloader.

    args:
        dataset: PyTorch dataset object with a .pirads attribute
        pirads_cutoff: PIRADS score cutoff
    returns:
        PyTorch WeightedRandomSampler object
    """

    # get the list of targets for the dataset
    targets = dataset.pirads

    # retrieve the sample weights
    sample_weights = compute_sample_weights(targets, pirads_cutoff)

    # return a WeightedRandomSampler
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(dataset), replacement=True
    )
    return sampler
