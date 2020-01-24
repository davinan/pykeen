# -*- coding: utf-8 -*-

"""Implementation of ConvE."""

import logging
import math
import sys
from typing import Optional, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from ..base import BaseModule, CUDNN_ERROR
from ..init import embedding_xavier_normal_
from ...losses import BCEAfterSigmoidLoss, Loss
from ...regularizers import Regularizer
from ...triples import TriplesFactory

__all__ = [
    'ConvE',
]

logger = logging.getLogger(__name__)


def _calculate_missing_shape_information(
    embedding_dim: int,
    input_channels: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    Automatically calculates missing dimensions for ConvE.

    :param embedding_dim:
    :param input_channels:
    :param width:
    :param height:

    :return: (input_channels, width, height), such that
            `embedding_dim = input_channels * width * height`
    :raises:
        If not factorization could be found.
    """
    # Store initial input for error message
    original = (input_channels, width, height)

    # All are None
    if all(factor is None for factor in [input_channels, width, height]):
        input_channels = 1
        result_sqrt = math.floor(math.sqrt(embedding_dim))
        height = max(factor for factor in range(1, result_sqrt + 1) if embedding_dim % factor == 0)
        width = embedding_dim // height

    # input_channels is None, and any of height or width is None -> set input_channels=1
    if input_channels is None and any(remaining is None for remaining in [width, height]):
        input_channels = 1

    # input channels is not None, and one of height or width is None
    assert len([factor for factor in [input_channels, width, height] if factor is None]) <= 1
    if width is None:
        width = embedding_dim // (height * input_channels)
    if height is None:
        height = embedding_dim // (width * input_channels)
    if input_channels is None:
        input_channels = embedding_dim // (width * height)
    assert not any(factor is None for factor in [input_channels, width, height])

    if input_channels * width * height != embedding_dim:
        raise ValueError(f'Could not resolve {original} to a valid factorization of {embedding_dim}.')

    return input_channels, width, height


class ConvE(BaseModule):
    """An implementation of ConvE from [dettmers2018]_.

    .. seealso::

        - Official Implementation: https://github.com/TimDettmers/ConvE/blob/master/model.py

    Example usage:

    >>> # Step 1: Get triples
    >>> from poem.datasets import Nations
    >>> dataset = Nations(create_inverse_triples=True)
    >>> # Step 2: Configure the model
    >>> from poem.models import ConvE
    >>> model = ConvE(
    ...     embedding_dim       = 200,
    ...     input_channels      = 1,
    ...     output_channels     = 32,
    ...     embedding_height    = 10,
    ...     embedding_width     = 20,
    ...     kernel_height       = 3,
    ...     kernel_width        = 3,
    ...     input_dropout       = 0.2,
    ...     feature_map_dropout = 0.2,
    ...     output_dropout      = 0.3,
    ...     preferred_device    = 'gpu',
    ... )
    >>> # Step 3: Configure the loop
    >>> from torch.optim import Adam
    >>> optimizer = Adam(params=model.get_grad_params())
    >>> from poem.training import LCWATrainingLoop
    >>> training_loop = LCWATrainingLoop(model=model, optimizer=optimizer)
    >>> # Step 4: Train
    >>> losses = training_loop.train(num_epochs=5, batch_size=256)
    >>> # Step 5: Evaluate the model
    >>> from poem.evaluation import RankBasedEvaluator
    >>> evaluator = RankBasedEvaluator()
    >>> metric_result = evaluator.evaluate(model=model, mapped_triples=dataset.testing.mapped_triples, batch_size=8192)
    """

    hpo_default = dict(
        input_channels=dict(type=int, low=1, high=3),
        output_channels=dict(type=int, low=16, high=64),
        embedding_height=dict(type=int, low=5, high=15),
        embedding_width=dict(type=int, low=15, high=25),
        kernel_height=dict(type=int, low=2, high=4),
        kernel_width=dict(type=int, low=2, high=4),
        input_dropout=dict(type=float, low=0.0, high=1.0),
        output_dropout=dict(type=float, low=0.0, high=1.0),
        feature_map_dropout=dict(type=float, low=0.0, high=1.0),
    )
    loss_default: Type[Loss] = BCEAfterSigmoidLoss
    loss_default_kwargs = {}

    def __init__(
        self,
        triples_factory: TriplesFactory,
        entity_embeddings: Optional[nn.Embedding] = None,
        relation_embeddings: Optional[nn.Embedding] = None,
        bias_term: Optional[nn.Parameter] = None,
        input_channels: Optional[int] = None,
        output_channels: int = 32,
        embedding_height: Optional[int] = None,
        embedding_width: Optional[int] = None,
        kernel_height: int = 3,
        kernel_width: int = 3,
        input_dropout: float = 0.2,
        output_dropout: float = 0.3,
        feature_map_dropout: float = 0.2,
        embedding_dim: int = 200,
        loss: Optional[Loss] = None,
        preferred_device: Optional[str] = None,
        random_seed: Optional[int] = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        """Initialize the model."""
        # ConvE should be trained with inverse triples
        if not triples_factory.create_inverse_triples:
            logger.warning(
                '\nThe ConvE model should be trained with inverse triples.\n'
                'This can be done by defining the TriplesFactory class with the _create_inverse_triples_ parameter set '
                'to true.',
            )

        super().__init__(
            triples_factory=triples_factory,
            embedding_dim=embedding_dim,
            entity_embeddings=entity_embeddings,
            loss=loss,
            preferred_device=preferred_device,
            random_seed=random_seed,
            regularizer=regularizer,
        )

        # Embeddings
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.bias_term = bias_term

        # Automatic calculation of remaining dimensions
        logger.info(f'Resolving {input_channels} * {embedding_width} * {embedding_height} = {embedding_dim}.')
        if embedding_dim is None:
            embedding_dim = input_channels * embedding_width * embedding_height

        # Parameter need to fulfil:
        #   input_channels * embedding_height * embedding_width = embedding_dim
        input_channels, embedding_width, embedding_height = _calculate_missing_shape_information(
            embedding_dim=embedding_dim,
            input_channels=input_channels,
            width=embedding_width,
            height=embedding_height,
        )
        logger.info(f'Resolved to {input_channels} * {embedding_width} * {embedding_height} = {embedding_dim}.')

        self.embedding_height = embedding_height
        self.embedding_width = embedding_width
        self.input_channels = input_channels

        if self.input_channels * self.embedding_height * self.embedding_width != self.embedding_dim:
            raise ValueError(
                f'Product of input channels ({self.input_channels}), height ({self.embedding_height}), and width '
                f'({self.embedding_width}) does not equal target embedding dimension ({self.embedding_dim})',
            )

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(output_dropout)
        self.feature_map_drop = torch.nn.Dropout2d(feature_map_dropout)

        self.conv1 = torch.nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=output_channels,
            kernel_size=(kernel_height, kernel_width),
            stride=1,
            padding=0,
            bias=True,
        )
        # num_features – C from an expected input of size (N,C,L)
        self.bn0 = torch.nn.BatchNorm2d(self.input_channels)
        # num_features – C from an expected input of size (N,C,H,W)
        self.bn1 = torch.nn.BatchNorm2d(output_channels)
        self.bn2 = torch.nn.BatchNorm1d(self.embedding_dim)
        num_in_features = \
            output_channels \
            * (2 * self.embedding_height - kernel_height + 1) \
            * (self.embedding_width - kernel_width + 1)
        self.fc = torch.nn.Linear(num_in_features, self.embedding_dim)

        # Finalize initialization
        self._init_weights_on_device()

    def init_empty_weights_(self):  # noqa: D102
        if self.entity_embeddings is None:
            self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
            embedding_xavier_normal_(self.entity_embeddings)
        if self.relation_embeddings is None:
            self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
            embedding_xavier_normal_(self.relation_embeddings)
        if self.bias_term is None:
            self.bias_term = nn.Parameter(torch.zeros(self.num_entities))

        return self

    def clear_weights_(self):  # noqa: D102
        self.entity_embeddings = None
        self.relation_embeddings = None
        self.bias_term = None
        return self

    def _convolve_entity_relation(self, h: torch.LongTensor, r: torch.LongTensor) -> torch.FloatTensor:
        """Perform the main calculations of the ConvE model."""
        batch_size = h.shape[0]

        # batch_size, num_input_channels, 2*height, width
        x = torch.cat([h, r], dim=2)

        try:
            # batch_size, num_input_channels, 2*height, width
            x = self.bn0(x)

            # batch_size, num_input_channels, 2*height, width
            x = self.inp_drop(x)
            # (N,C_out,H_out,W_out)
            x = self.conv1(x)

            x = self.bn1(x)
            x = F.relu(x)
            x = self.feature_map_drop(x)
            # batch_size, num_output_channels * (2 * height - kernel_height + 1) * (width - kernel_width + 1)
            x = x.view(batch_size, -1)
            x = self.fc(x)
            x = self.hidden_drop(x)

            if batch_size > 1:
                x = self.bn2(x)
            x = F.relu(x)
        except RuntimeError as e:
            if e.args[0] == CUDNN_ERROR:
                logger.warning(
                    '\nThis code crash might have been caused by a CUDA bug, see '
                    'https://github.com/allenai/allennlp/issues/2888, '
                    'which causes the code to crash during evaluation mode.\n'
                    'To avoid this error, the batch size has to be reduced.\n'
                    f'The original error message: \n{e.args[0]}',
                )
                sys.exit(1)
            else:
                raise e
        return x

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(hrt_batch[:, 0]).view(
            -1,
            self.input_channels,
            self.embedding_height,
            self.embedding_width,
        )
        r = self.relation_embeddings(hrt_batch[:, 1]).view(
            -1,
            self.input_channels,
            self.embedding_height,
            self.embedding_width,
        )
        t = self.entity_embeddings(hrt_batch[:, 2])

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        x = self._convolve_entity_relation(h, r)

        # For efficient calculation, each of the convolved [h, r] rows has only to be multiplied with one t row
        x = (x.view(-1, self.embedding_dim) * t).sum(dim=1, keepdim=True)

        """
        In ConvE the bias term add the end is added for each tail item. In the OWA assumption we only have one tail item
        for each head and relation. Accordingly the relevant bias for each tail item and triple has to be looked up.
        """
        x += self.bias_term[hrt_batch[:, 2, None]]
        # The application of the sigmoid during training is automatically handled by the default loss.

        return x

    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        h = self.entity_embeddings(hr_batch[:, 0]).view(
            -1,
            self.input_channels,
            self.embedding_height,
            self.embedding_width,
        )
        r = self.relation_embeddings(hr_batch[:, 1]).view(
            -1,
            self.input_channels,
            self.embedding_height,
            self.embedding_width,
        )
        t = self.entity_embeddings.weight.transpose(1, 0)

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        x = self._convolve_entity_relation(h, r)

        x = x @ t
        x += self.bias_term.expand_as(x)
        # The application of the sigmoid during training is automatically handled by the default loss.

        return x

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102
        rt_batch_size = rt_batch.shape[0]
        h = self.entity_embeddings.weight
        r = self.relation_embeddings(rt_batch[:, 0]).view(
            -1,
            self.input_channels,
            self.embedding_height,
            self.embedding_width,
        )
        t = self.entity_embeddings(rt_batch[:, 1])

        # Embedding Regularization
        self.regularize_if_necessary(h, r, t)

        '''
        Every head has to be convolved with every relation in the rt_batch. Hence we repeat the
        relation _num_entities_ times and the head _rt_batch_size_ times.
        '''
        r = r.repeat(h.shape[0], 1, 1, 1)
        # Code to repeat each item successively instead of the entire tensor
        h = h.unsqueeze(1).repeat(1, rt_batch_size, 1).view(
            -1,
            self.input_channels,
            self.embedding_height,
            self.embedding_width,
        )

        x = self._convolve_entity_relation(h, r)

        '''
        For efficient computation, each convolved [h, r] pair has only to be multiplied with the corresponding t
        embedding found in the rt_batch with [r, t] pairs.
        '''
        x = (x.view(self.num_entities, rt_batch_size, self.embedding_dim) * t[None, :, :]).sum(2).transpose(1, 0)

        """
        In ConvE the bias term at the end is added for each tail item. In the score_h function, each row holds
        the same tail for many different heads, meaning that these items have to be looked up for each tail of each row
        and only then can be added correctly.
        """
        x += self.bias_term[rt_batch[:, 1, None]]
        # The application of the sigmoid during training is automatically handled by the default loss.

        return x
