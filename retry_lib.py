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

"""Library for running LLM calls with retries."""

import logging
import time
from typing import Callable, TypeVar


T = TypeVar('T')


def call_function_with_retry(
    function: Callable[[], T],
    num_attempts: int = 1,
) -> T:
  """Calls a function and retries if a StatusNotOk exception is raised.

  Uses exponential back-off.

  Args:
    function: The function that performs the RPC call.
    num_attempts: Number of times to retry if the request raises an exception
      (e.g. due to timeout). If None, retries indefinitely.

  Raises:
    ConnectionError: if RPC fails.
  Returns:
    The return value of the RPC call or None if all attempts fail.
  """
  attempt_index = 0
  error_message = ''
  while num_attempts is None or attempt_index < num_attempts:
    try:
      return function()
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error(
          'Failed, preparing for %d-th retry ...',
          attempt_index + 1,
          exc_info=True,
      )
      error_message = str(e)
      if num_attempts is None or attempt_index + 1 < num_attempts:
        time.sleep(2**attempt_index)
    attempt_index += 1
  raise ConnectionError(
      f'RPC to LLM failed. Last error message {error_message}'
  )
