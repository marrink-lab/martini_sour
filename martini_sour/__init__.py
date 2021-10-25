# Copyright 2021 University of Groningen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pbr.version
#from .log_helpers import StyleAdapter, get_logger

__version__ = pbr.version.VersionInfo('martini_sour').release_string()

# Find the data directory once.
try:
    import pkg_resources
except ImportError:
    import os
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
    TEST_DATA = os.path.join(os.path.dirname(__file__), 'tests/test_data')
    del os
else:
    DATA_PATH = pkg_resources.resource_filename('martini_sour', 'data')
    TEST_DATA = pkg_resources.resource_filename('martini_sour', 'tests/test_data')
    del pkg_resources

del pbr

from .src.conv_coords import conv_coords
from .src.conv_itp import conv_itp
from .src.analyze import analyze
from .src.titrate import titrate