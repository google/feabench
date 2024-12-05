# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common file parsing utilities."""

import numpy as np


JAVA_DELIMITER = ';'
JAVA_PADDER = ';'
TEX_DELIMITER = '\n'
TEX_PADDER = ''


def split_string_by_delimiter(
    string: str, delimiter: str, padding: str = ''
) -> np.ndarray:
  """Returns array of lines given string split on delimiter, with padding added."""
  lines = np.array(
      [f + padding for f in string.split(delimiter) if f], dtype=str
  )
  return lines

