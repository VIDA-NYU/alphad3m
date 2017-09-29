from __future__ import division

from vistrails.core.modules.utils import make_modules_dict

from .readdataset import _modules as read_dataset_modules
from .readimage import _modules as read_image_file

_modules = make_modules_dict(read_dataset_modules, read_image_file,
                             namespace = 'read')
