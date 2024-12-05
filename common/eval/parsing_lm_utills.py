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

"""Functions related to parsing the output of a language model."""
# pylint: disable=broad-exception-caught

import json
import re
from typing import Any, Callable, NotRequired, Optional, Sequence, TypedDict
import warnings

import numpy as np
import numpy.typing as npt

from engmod import common_parsing_utils
from engmod import parse_java_api


class CodeBlock:
  """Holds the code and the pythonized code."""

  def __init__(self, code: Sequence[str]):
    self.code = []
    self.pythonized_code = []
    for line in code:
      try:
        pyline = parse_java_api.pythonize_java_api(line)
        self.code.append(line)
        self.pythonized_code.append(pyline)
      except (ValueError, SyntaxError) as e:
        warnings.warn(f"Can't pythonize: {line}, {str(e)}")
    # removed an outer except block here since it breaks looped evals.


LMSolution = TypedDict(
    """LM Solution. Every `code state` is an LMSolution.

    Attributes:
      lm_code_reply: The raw, NON-parsed reply from the LM, with
        OUTPUT_PATH/output.txt. (for LM visibility) Might contain comments
        that aren't code.
      target_path: The path to the target file.
      ParsingSuccessful: Whether the parsing was successful.
      CodeBlock: The parsed code block with actual target_path (for execution).
        This should almost always be there. Only absent if ParsingSuccessful is
        False.
      exported_table: The table if exported by the LM.
      prompt: The prompts used to generate the reply. Only used in
        SingleStepMultiLMAgent.
    """ 'LMSolution',
    {
        'lm_code_reply': str,
        'target_path': str,
        'ParsingSuccessful': bool,
        'CodeBlock': NotRequired[CodeBlock],
        'exported_table': NotRequired[str],
        'prompt': NotRequired[str],
    },
)


class Parser:
  """Implement your desired parsing strategy."""

  def __init__(self):
    pass

  def parse(
      self, lm_reply: str, target_path: str = 'OUTPUT_PATH/output.txt'
  ) -> dict[str, Any]:
    """Parses the raw LM output into an executable CodeBlock."""
    raise NotImplementedError()


class CodeParser(Parser):
  """Parses the raw LM output into an executable CodeBlock."""

  def __init__(self, postproc_fn: Callable[[str], np.ndarray]):
    self.postproc_fn = postproc_fn

  def parse(
      self, lm_reply: str, target_path: str = 'OUTPUT_PATH/output.txt'
  ) -> LMSolution:
    """Parses the raw LM output into an executable CodeBlock.

    Args:
      lm_reply: The raw LM output, mostly code, but may also have other content.
      target_path: The path to the target file.

    Returns:
      LMSolution
    """
    try:
      lm_exec = lm_reply.replace('OUTPUT_PATH/output.txt', target_path)
      lm_code = self.postproc_fn(lm_exec)
      return LMSolution(
          lm_code_reply=lm_reply,
          target_path=target_path,
          ParsingSuccessful=True,
          CodeBlock=CodeBlock(lm_code),
      )
    except Exception as e:
      print(
          'Parser: Could not parse reply into code, returning existing code: '
          + lm_reply
          + str(e)
      )
      return LMSolution(
          lm_code_reply=lm_reply,
          target_path=target_path,
          ParsingSuccessful=False,
      )

  def unparse(self, code: CodeBlock, target_path: Optional[str] = None) -> str:
    """Unparses the CodeBlock into a string."""
    code = '\n'.join(code.code)
    if target_path:
      code = code.replace(target_path, 'OUTPUT_PATH/output.txt')
    return code


# Parsing Strategies
def filter_code(lm_reply: str) -> str:
  """Earlier postprocessing strategy to get code.

  Args:
    lm_reply: LM output string

  Returns:
  """
  start_tag = '```java'
  end_tag = '```[^j]'
  # get indices
  print('Start tag:')
  start_pos = []
  for match in re.finditer(start_tag, lm_reply):
    start_pos.append(match.end())
  print('End tag:')
  end_pos = []
  for match in re.finditer(end_tag, lm_reply):
    end_pos.append(match.start())
  assert len(start_pos) == len(end_pos), print(len(start_pos), len(end_pos))

  code_str = ''
  for start, end in zip(start_pos, end_pos):
    code_str += lm_reply[start:end]
  return code_str


def postprocess_result(code: str) -> npt.NDArray[str] | None:
  r"""Converts a code block to a list of code lines.

  This ensures all lines of code must start with model. and end with ;.
  Assumptions: 1. There's a single contiguous block of java code.
  2. Every valid line must start with model. and end with ;.
  3. There may be lines where a single line of code is split by a \n, this
  handles those.

  Args:
    code: LM output string.

  Returns:
    An array of code lines.
  """
  try:
    code = code[
        code.index('model.') : code.rindex(';') + 1
    ]  # because it'll arbitrarily return things like "java```"
    # ensures the code block as a whole starts and ends with "model." and ";"
    answer_parsed = common_parsing_utils.split_string_by_delimiter(
        code,
        common_parsing_utils.JAVA_DELIMITER,
        common_parsing_utils.JAVA_PADDER,
    )  # splits on ;
    answer_parsed = [
        c.replace('\n', '') for c in answer_parsed
    ]  # for JAVA lines that were split into multiple lines
    pat = re.compile(r'model\..*;')
    final_answer_parsed = []
    for l in answer_parsed:
      match = re.search(pat, l)
      if match:
        final_answer_parsed.append(match.group())
    return np.array(final_answer_parsed)

  except ValueError as e:
    print(e, code)


def parse_json_output_to_dict(output: str) -> dict[str, Any] | None:
  """Parses the output of the LM into a JSON dictionary."""
  try:
    start = output.rindex('{')
    end = output.rindex('}')
    return json.loads(output[start : end + 1])
  except (ValueError, json.JSONDecodeError):
    return None
