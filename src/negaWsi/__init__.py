"""Compute Module"""

from negaWsi.alternating_methods.side_info.imc import IMC
from negaWsi.alternating_methods.standard.agd import AGD
from negaWsi.alternating_methods.standard.lsq import ALSQ
from negaWsi.alternating_methods.standard.scipy_minimize import AMD
from negaWsi.side_info.enega_fs import ENegaFS
from negaWsi.side_info.nega_fs import NegaFS
from negaWsi.side_info.nega_reg import NegaReg
from negaWsi.standard.nega import Nega
from negaWsi.utils.result import Result

__all__ = [
    "Nega",
    "ENegaFS",
    "NegaFS",
    "NegaReg",
    "Result",
    "IMC",
    "AGD",
    "ALSQ",
    "AMD",
]
