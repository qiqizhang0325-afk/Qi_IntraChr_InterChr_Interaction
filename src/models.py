"""Models for intra-chromosome and inter-chromosome SNP interaction analysis.

This module provides backward compatibility by importing models from separate files.
For new code, prefer importing directly from intra_chr_model or inter_chr_model.
"""

# Import models from separate files for better code organization
try:
    from .intra_chr_model import IntraChrModel
    from .inter_chr_model import InterChrModel
except ImportError:
    # Fallback for direct execution
    from intra_chr_model import IntraChrModel
    from inter_chr_model import InterChrModel

# Export for backward compatibility
__all__ = ['IntraChrModel', 'InterChrModel']
