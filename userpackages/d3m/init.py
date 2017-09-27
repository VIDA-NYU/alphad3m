from __future__ import division

from vistrails.core.modules.utils import make_modules_dict

from .primitives import _modules as primitives_modules
from .read import _modules as read_modules


_modules = [primitives_modules, read_modules]
_modules = make_modules_dict(*_modules)
