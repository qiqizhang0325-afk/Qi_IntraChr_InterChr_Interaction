"""QI Intra/Inter Chromosome Interaction Analysis Package."""

from .data_processor import VCFProcessor
from .dataset import SNPDataset
from .model_components import BiMambaBlock, PositionalEncoding
from .models import InterChrModel, IntraChrModel
from .training import integrate_results, train_model

__all__ = [
    'VCFProcessor',
    'SNPDataset',
    'BiMambaBlock',
    'PositionalEncoding',
    'IntraChrModel',
    'InterChrModel',
    'train_model',
    'integrate_results',
]

__version__ = '1.2.0'



