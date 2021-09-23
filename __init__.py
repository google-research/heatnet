# Copyright 2021 Google LLC
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
"""HeatNet is a python package for heat wave prediction using deep learning."""

__version__ = '0.1.0'

# Allow `import heatnet`; `heatnet.data.[...]`, etc
from . import data
from . import evaluation
from . import model
from .file_util import copy_dir
from .file_util import ext_to_local
from .file_util import load_dataset
from .file_util import save_dict_to_json
