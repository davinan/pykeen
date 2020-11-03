# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np
import torch

from .perturbator import Perturbator
from ..triples import TriplesFactory

__all__ = [
    'RandomPerturbator',
]


class RandomPerturbator(Perturbator):

    #: The default strategy for optimizing the negative sampler's hyper-parameters
    hpo_default = dict(
        num_negs_per_pos=dict(type=int, low=1, high=100, q=10),
    )

    def __init__(
            self,
            triples_factory: TriplesFactory,
            deletion_ratio: Optional[float] = 0.05,
            neg_ratio: Optional[float] = 0.04,
    ) -> None:
        super().__init__(
            triples_factory=triples_factory,
        )
        self.deletion_ratio = deletion_ratio
        self.neg_ratio = neg_ratio

    def sample(self, positive_batch: torch.LongTensor) -> torch.LongTensor:
        batch_size = positive_batch.shape[0]

        num_keep = batch_size * (1 - self.deletion_ratio)
        num_neg = int(batch_size * self.neg_ratio)

        indices2keep = np.random.choice(
            np.arange(batch_size), int(num_keep), replace=False
        )

        positive_batch = torch.index_select(
            positive_batch, 0, torch.tensor(indices2keep)
        )

        # shuffle the batch
        positive_batch = positive_batch[torch.randperm(positive_batch.size()[0])]
        positive_batch2perturb = positive_batch[:num_neg]

        # Bind number of negatives to sample
        num_negs = positive_batch2perturb.shape[0]

        # Equally corrupt head and tail
        split_idx = num_negs // 2

        # Copy positive batch for corruption.
        # Do not detach, as no gradients should flow into the indices.
        negative_batch = positive_batch2perturb.clone()

        # Sample random entities as replacement
        negative_entities = torch.randint(high=self.num_entities - 1, size=(num_negs,), device=positive_batch.device)

        # Replace heads – To make sure we don't replace the head by the original value
        # we shift all values greater or equal than the original value by one up
        # for that reason we choose the random value from [0, num_entities -1]
        filter_same_head = (negative_entities[:split_idx] >= positive_batch2perturb[:split_idx, 0])
        negative_batch[:split_idx, 0] = negative_entities[:split_idx] + filter_same_head.long()
        # Corrupt tails
        filter_same_tail = (negative_entities[split_idx:] >= positive_batch2perturb[split_idx:, 2])
        negative_batch[split_idx:, 2] = negative_entities[split_idx:] + filter_same_tail.long()

        positive_batch[:num_neg] = negative_batch

        return positive_batch
