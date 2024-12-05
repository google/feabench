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

"""Parse tutorial code."""

import re
from typing import Sequence

import dataclasses

_MAIN_FN_DEFINITION_START = "  public static void main(String[] args) {\n"
_FN_DEFINITION_END = "\n  }"
_CREATE_MODEL = 'Model model = ModelUtil.create("Model");\n'
_RETURN = "return model;\n"

_RUN_FNS_PATTERN = r"(?P<fn_name>[a-z0-9]*)\([a-z]*\);"

_FN_DEF_PATTERN = r"  public static Model %s\((Model model)?\) \{\n"

_MODEL_ATTR_TEMPLATE = r"^\s*model\.%s\(\"[^;]*;\n"
_MODEL_ATTR_NAMES = (
    "modelPath",
    "title",
    "description",
    "label",
)


@dataclasses.dataclass(frozen=True)
class RunFunction:
  """Encapsulates a single .run() function in the JAVA code."""

  name: str = ""
  body: str = ""


@dataclasses.dataclass(frozen=True)
class TutorialCode:
  """Tutorial code."""

  model_id: int = -1
  version: str = ""
  source_path: str = ""
  raw_code: str = ""
  main_fn_body: str = ""
  run_function: Sequence[RunFunction] = dataclasses.field(default_factory=list)


def _extract_main_fn_body(java_code: str) -> str:
  """Extracts the code contents of the main function.

  Args:
    java_code: The API code extracted from a tutorial model mph binary. This is
      machine generated, and should conform to expected patterns. 

  Returns:
    A string containing the contents of the main function.
  """
  # Get the start index.
  main_start_find_idx = java_code.find(_MAIN_FN_DEFINITION_START)

  if main_start_find_idx == -1:
    raise ValueError("No main function found.")
  main_start_idx = main_start_find_idx + len(_MAIN_FN_DEFINITION_START)

  # Get the end index.
  main_end_find_idx = java_code[main_start_idx:].find(_FN_DEFINITION_END)

  if main_end_find_idx == -1:
    raise ValueError("No end to main function found.")
  return java_code[main_start_idx : main_start_idx + main_end_find_idx] + "\n"


def _get_run_functions(main_fn_body: str) -> list[str]:
  """Extracts the names of run functions from the main function body.

  "Run functions" contain the execution logic of a model.

  Args:
    main_fn_body: The body of the main function.

  Returns:
    A list of run function names.
  """
  return re.findall(_RUN_FNS_PATTERN, main_fn_body)


def _get_public_function_body(java_code: str, fn_name: str) -> str:
  """Extracts the code contents of a public function.

  Args:
    java_code: a string from a tutorial. Inputs are filtered to conform to expected patterns.
    fn_name: the name of the function to extract.

  Returns:
    The body of the named function.
  """
  # Using % format because f-strings are poorly supported with regex.
  fn_def_pattern = _FN_DEF_PATTERN % fn_name
  search_result = re.search(fn_def_pattern, java_code)
  if not search_result:
    raise ValueError(f"No function definition found for {fn_name}.")
  start_idx = search_result.end()

  remaining_code = java_code[start_idx:]
  end_idx = remaining_code.find(_FN_DEFINITION_END)
  if end_idx == -1:
    raise ValueError(f"No function body end found for {fn_name}.")

  return remaining_code[:end_idx] + "\n"


def _cleanup_java_formatting(java_code: str) -> str:
  """Formats to one expression per line, removing blank lines, comments, and leading white space."""
  code = java_code.replace(";\n", ";").replace(";", ";\n")  # one exp per line
  cleaned_lines = []
  for line in code.splitlines(True):  # keep '\n' line ending
    if not line:
      continue  # remove blank lines
    if line.startswith("//"):
      continue  # remove commented lines
    line = line.lstrip()  # remove leading whitespace
    # undo line continuations
    line = line if line.endswith(";\n") else line.rstrip("\n")
    cleaned_lines.append(line)
  return "".join(cleaned_lines)


def _clean_fn_body(fn_body: str) -> str:
  """Standardizes formatting and removes specific lines from a function body.

  Args:
    fn_body: the code body of a function.

  Returns:
    A cleaned version of the function body.
  """
  # Clean formatting.
  cleaned_code = _cleanup_java_formatting(fn_body)
  # Remove return statement.
  cleaned_code = cleaned_code.replace(_RETURN, "")
  # Remove create model statement.
  cleaned_code = cleaned_code.replace(_CREATE_MODEL, "")

  # Remove identifying metadata assignments.
  for attr_name in _MODEL_ATTR_NAMES:
    attr_pattern = _MODEL_ATTR_TEMPLATE % attr_name
    cleaned_code = re.sub(attr_pattern, "", cleaned_code, flags=re.MULTILINE)

  return cleaned_code


def parse_java(
    java_code: str,
) -> TutorialCode:
  """Given model java code, parse it into components.

  The input is machine generated, so we expect it to conform to a pattern.

  Args:
    java_code: java code from a tutorial. Inputs are filtered to conform
      to expected patterns.

  Returns:
     TutorialCode: Contains the structured components of the code.
  """
  main_fn_body = _clean_fn_body(_extract_main_fn_body(java_code))

  run_fn_names = _get_run_functions(main_fn_body)
  run_fn_bodies = [
      _clean_fn_body(_get_public_function_body(java_code, fn_name))
      for fn_name in run_fn_names
  ]

  return TutorialCode(
      main_fn_body=main_fn_body,
      run_function=[
          RunFunction(name=fn_name, body=fn_body)
          for fn_name, fn_body in zip(run_fn_names, run_fn_bodies)
      ],
  )
