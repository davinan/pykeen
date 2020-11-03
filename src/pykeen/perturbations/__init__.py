# -*- coding: utf-8 -*-

"""KG augmentation.
"""

from typing import Mapping, Set, Type, Union

from .random_perturbator import RandomPerturbator
from .perturbator import Perturbator
from .no_perturbator import NoPerturbator
from ..utils import get_cls, normalize_string

__all__ = [
    'random_perturbator.py',
    'perturbator.py',
    'perturbators',
    'get_perturbator_cls',
]

_PERTURBATOR_SUFFIX = 'PERTURBATOR'
_PERTURBATORS: Set[Type[Perturbator]] = {
    RandomPerturbator,
}

#: A mapping of negative samplers' names to their implementations
perturbators: Mapping[str, Type[Perturbator]] = {
    normalize_string(cls.__name__, suffix=_PERTURBATOR_SUFFIX): cls
    for cls in _PERTURBATORS
}


def get_perturbator_cls(query: Union[None, str, Type[Perturbator]]) -> Type[Perturbator]:
    """Get the negative sampler class."""
    return get_cls(
        query,
        base=Perturbator,
        lookup_dict=perturbators,
        default=NoPerturbator,
        suffix=_PERTURBATOR_SUFFIX,
    )
