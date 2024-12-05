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

"""COMSOL executability evaluation utils."""

from typing import Any, Sequence
import warnings

import numpy as np
import termcolor

from engmod import file_utils
from engmod.common.eval import parsing_lm_utils
from engmod.common.remote_service import mph_comsol_client


def pass_code_linewise_to_client(
    lines: Sequence[str], client: mph_comsol_client.MphComsolClient
) -> list[Any]:
  """Sends a list of pythonized COMSOL API calls to the client and returns replies."""
  replies = []
  for i, line in enumerate(lines):
    reply = str(client.model_api_call(line))
    print(f'Line {i}:{line} -> {reply}')
    replies.append(reply)
  return replies


def render_code_by_correctness(
    code: Sequence[str], mph_replies: Sequence[str]
) -> list[str]:
  """Classifies each line of code based on the return message upon execution.

  Warning: Errors that do not follow the patterns listed here will be considered
  "correct".

  Args:
    code: Sequence of code operations
    mph_replies: Sequence of messages from the COMSOL-MPH service.

  Returns:
    List of flags where flag = Correct | Syntax error | Translation error
  """

  syntax_error_flags = [
      'Messages',
      'has no attribute',
      'No matching overloads',
      'No objects',
      'invalid syntax',
      'Exception',
      'is not defined',
  ]
  # TODO: 'is not defined',  # Eg: model.component("comp1").geom("geom1").feature(
  # "wp_h" + i).geom().feature("r_h" + i).set("pos", ["0", "0"]) -> name 'i'
  # is not defined
  translation_error_flags = ['Ambiguous', 'comma', 'No Model set']
  # No model set is technically an error that arises if we run with a broken
  # connection.
  cdict = {
      'Correct': 'green',
      'Syntax error': 'red',
      'Translation error': 'magenta',
  }
  flags = []
  for l, reply in zip(code, mph_replies):
    if any(s in reply for s in syntax_error_flags):
      flag = 'Syntax error'
    elif any([s in reply for s in translation_error_flags]):
      flag = 'Translation error'
    else:
      flag = 'Correct'
    out = termcolor.colored(l, cdict[flag])
    print(out)
    flags.append(flag)
  return flags


def reinitialize_client(
    client: mph_comsol_client.MphComsolClient,
    path: str | None = None,
    model_name: str = 'Untitled',
    previous_code: Sequence[str] | None = None,
) -> None:
  """Reinitializes the client.

  Args:
    client: MphComsolClient
    path: Path to the model. This might be a model that was just created and
      saved or some with previously run lines of code built.
    model_name: Name of the model to create if path is not given.
    previous_code: Lines of code to run before the current code if path is not
      given.

  Returns:
    None
  """
  if path:
    assert path.endswith('.mph'), 'Must load an Mph file.'
    client.clear_models()
    client.load_model(path)
    print(f'Loaded model {client.model_name()}')
  elif previous_code:
    client.clear_models()
    client.create_model(model_name)
    assert client.model_name() == model_name
    past_replies = pass_code_linewise_to_client(previous_code, client)
    past_flags = render_code_by_correctness(previous_code, past_replies)
    # count number of errors in past flags
    num_errors = past_flags.count('Syntax error') + past_flags.count(
        'Translation error'
    )
    warnings.warn(f'Warning! past code had {num_errors} errors.')
  else:
    # No previous code, just create a new model.
    client.clear_models()
    client.create_model(model_name)
    assert client.model_name() == model_name
  return


def test_code_in_client(
    client: mph_comsol_client.MphComsolClient,
    code: Sequence[str],
    num_lines: int | None = None,
) -> tuple[list[str], list[str]]:
  """Loads the model at a current state and tests an incremental code snippet.

  Args:
    client: MphComsolClient
    code: Lines of code to test. pythonize_java_api should have been run
      externally.
    num_lines: Number of lines to test

  Returns:
    llm_replies: List of replies from the model
    flags: List of flags indicating the correctness of each line of code.
  """
  if num_lines:
    code = code[: min(num_lines, len(code))]
  llm_replies = pass_code_linewise_to_client(code, client)
  flags = render_code_by_correctness(code, llm_replies)
  return llm_replies, flags


def jsonize(value: Any) -> Any:
  """Converts a value to a json-serializable value."""
  if isinstance(value, str) or isinstance(value, list):
    return value
  elif isinstance(value, float):
    return str(value)
  elif isinstance(value, np.ndarray):
    return value.tolist()
  elif (
      isinstance(value, int)
      or isinstance(value, np.int64)
      or isinstance(value, np.int32)
  ):
    return int(value)
  elif isinstance(value, parsing_lm_utils.CodeBlock):
    return jsonize(value.code)
  else:
    print(f'Unsupported type for {value}: {type(value)}')
    return None


def compute_relative_error(value: float, target_value: float) -> float:
  """Computes the relative error between the predicted value and the target value."""
  return abs(value - target_value) / abs(target_value)
