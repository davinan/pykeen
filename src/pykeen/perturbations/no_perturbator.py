# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np
import torch

from .perturbator import Perturbator
from ..triples import TriplesFactory

__all__ = [
    'NoPerturbator',
]


class NoPerturbator(Perturbator):

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
        return positive_batch
