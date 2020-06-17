import os
import pkg_resources

__version__ = pkg_resources.require('monet')[0].version

_root = os.path.abspath(os.path.dirname(__file__))

from .core import ExpMatrix
from .latent import MonetModel
from .denoise import ENHANCE
