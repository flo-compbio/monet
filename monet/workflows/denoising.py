# Copyright (c) 2021 Florian Wagner
#
# This file is part of Monet.

from typing import Union
import logging

from ..core import ExpMatrix
from ..denoise import EnhanceModel

_LOGGER = logging.getLogger(__name__)


def enhance(expression_file: str, output_file: str = None,
            num_components: Union[str, int] = 'mcv') -> EnhanceModel:

    matrix = ExpMatrix.load(expression_file)

    enhance_model = EnhanceModel(num_components=num_components)
    enhance_model.fit(matrix)

    if output_file is not None:
        enhance_model.save_pickle(output_file)

    return enhance_model
