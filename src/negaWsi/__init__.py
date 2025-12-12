"""Compute Module"""

from negaWsi.nega import Nega
from negaWsi.nega_fs import NegaFS
from negaWsi.enega_fs import ENegaFS
from negaWsi.nega_reg import NegaReg
from negaWsi.result import Result
from negaWsi.comparative.imc import IMC

__all__ = ["Nega", "ENegaFS", "NegaFS", "NegaReg", "Result", "IMC"]
