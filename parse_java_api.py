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

"""Tools for parsing COMSOL API."""

import ast
import re
from typing import Any, Optional


def separate_run_method_body(code: str) -> dict[str, str]:
  """Separate body of the run() method from everything before, after."""
  run_declaration = 'public static Model run() {\n'
  split = code.split(run_declaration)
  assert len(split) == 2
  before = split[0] + run_declaration
  run_closing = 'return model;\n'
  second_split = split[1].split(run_closing)
  assert len(second_split) > 1  # possible to have auxiliary run.*() methods
  body = second_split[0] + run_closing
  after = run_closing.join(second_split[1:])
  assert before + body + after == code
  return {'before': before, 'body': body, 'after': after}


def compactify_java(code: str) -> tuple[str, dict[str, str]]:
  """Remove whitespace and specific lines unnecessary for direct API calls."""
  # Make all java statement terminations include a newline.
  code = code.replace(';\n', ';').replace(';', ';\n')
  lines = code.splitlines(True)  # keep '\n' line ending
  compact_lines = []
  removed_lines = {}
  for line in lines:
    if 'Model model = ModelUtil.create(' in line:
      removed_lines['model_creation'] = line
      continue  # Remove Model object creation.
    if 'model.modelPath(' in line:
      removed_lines['model_path'] = line
      continue  # Remove model path specification.
    if 'return model' in line:
      removed_lines['return_model'] = line
      continue  # Remove 'return model' at the end.
    if not line:
      continue  # Remove blank lines.
    line = line.lstrip()  # Remove leading whitespace.
    # undo line continuations
    line = line if line.endswith(';\n') else line.rstrip('\n')
    compact_lines.append(line)
  return ''.join(compact_lines), removed_lines


def _encountered_boolean_exceptions(argstring: str) -> bool:
  """Check if the argstring has exceptions that require conversion to "0" or "1"."""
  return 'RepeatLayerInMultilayerFilms' in argstring


def _get_api_argstring(line: str) -> str:
  """Returns the argument string of the rightmost method in the code line."""
  if not line.endswith(';'):
    raise ValueError(f'Expected java line to end with `;`, got {line}')
  m = re.fullmatch(r'.*\.\w+\((?P<args>.*)\);', line)
  if m is None:
    raise ValueError(f'Could not match args on last function of {line}')
  return m['args']


def _convert_arg_by_type(
    arg: Any, convert_overloaded_types: bool, argstring: Optional[str] = None
) -> str:
  """Convert java args to MPh python compatible args.

  Args:
    arg: the java function argument being converted.
    convert_overloaded_types: If true, modify the type conversion to resolve
      issues with MPh's python-wrapped overloaded java functions.
    argstring: the full argstring of the java function. adding this to check and
      handle exceptions differently.

  Returns:
    The converted arg, cast to a string for the purpose of constructing an
    MPh API string.
  """
  if isinstance(arg, str):
    return f'"{arg}"'
  elif convert_overloaded_types and isinstance(arg, bool):
    if argstring and _encountered_boolean_exceptions(argstring):
      return '"1"' if arg else '"0"'
    else:
      return '"on"' if arg else '"off"'
  elif convert_overloaded_types and isinstance(arg, (int, float)):
    return f'"{arg}"'
  elif isinstance(arg, list):
    args = [_convert_arg_by_type(x, convert_overloaded_types) for x in arg]
    return '[' + ', '.join(args) + ']'
  elif isinstance(arg, tuple):
    args = [_convert_arg_by_type(x, convert_overloaded_types) for x in arg]
    return '(' + ', '.join(args) + ')'
  else:
    return str(arg)


def _pythonize_argstring(argstring: str, convert_overloaded_types: bool) -> str:
  """Reformat java argstring to MPh-compatible python.

  The python-wrapped java objects in MPh create issues when calling an
  overloaded java function with python arguments where type conversion is
  ambiguous.  The `convert_overloaded_types` option is for resolving
  "ambiguous overload" errors.

  Example:
    .set(true, new int{1, 2, 3}) -> .set("on", ["1", "2", "3"])

  See unit tests for additional examples.

  Args:
    argstring: The string of arguments inside a java function expression.
    convert_overloaded_types: If true, modify the type conversion to resolve
      issues with MPh's python-wrapped overloaded java functions.
  Returns:
    The converted argstring, which can be replaces the java argstring to create
    an MPh-compatible API call.
  """
  if not argstring:  # Handle empty argstring.
    return argstring
  argstring = re.sub(r'new (int|float|double|String)(\[\])+', '', argstring)
  argstring = argstring.replace('{', '[').replace('}', ']')
  argstring = argstring.replace('true', 'True').replace('false', 'False')
  argstring = argstring.replace('Double.POSITIVE_INFINITY', '"inf"')
  argstring = argstring.replace('Double.NEGATIVE_INFINITY', '"-inf"')

  args = ast.literal_eval(argstring)
  # Handle single argument case.
  if isinstance(args, (str, bool, int, float)):
    # Handle case where set() has a single integer arg.
    if not isinstance(args, bool) and isinstance(args, int):
      convert_overloaded_types = False  # Keep integer types.
    return _convert_arg_by_type(args, convert_overloaded_types)
  # Handle case where set() args are all integers.
  if all(not isinstance(x, bool) and isinstance(x, int) for x in args):
    convert_overloaded_types = False  # Keep integer types.
  args = [
      _convert_arg_by_type(x, convert_overloaded_types, argstring) for x in args
  ]
  return ', '.join(args)


def pythonize_java_api(line: str) -> str:
  """Translate a line of java COMSOL API to be compatible with MPh."""
  argstring = _get_api_argstring(line)
  if not argstring:
    return line.rstrip(';')
  # The set() method requires additional type conversions to prevent an
  # "ambiguous overloads" error message from the java runtime environment.
  convert_overloaded_types = (
      '.set(' in line
  )
  end = f'({argstring});'
  start = line[: -len(end)]
  assert start + end == line, (start + end, line)
  argstring = _pythonize_argstring(argstring, convert_overloaded_types)
  # Exceptions to handle argstring. We probably want to refactor this module but
  # are taking the safer route of not making potentially breaking changes.
  if '.setIndex(' in line:
    args = ast.literal_eval(argstring)
    arg_types = [type(x) for x in args]
    if arg_types == [str, int, int, int] or arg_types == [str, int, int]:
      args_cast = [str(x) for x in args]
      args_cast[0] = f'"{args[0]}"'  # explicit string demarcators for str
      args_cast[1] = f'{args[1]}.0'
      # replaces 1 with 1.0 if the `value` is also an int.
      argstring = ', '.join(args_cast)
    elif arg_types[1] == bool and args[0] == 'FreeRotationAround':
      args_cast = [str(x) for x in args]
      args_cast[0] = f'"{args[0]}"'  # explicit string demarcators for str
      args_cast[1] = '1.0' if args[1] else '0.0'
      # replaces True with 1.0 and False with 0.0
      argstring = ', '.join(args_cast)

  # Add back to line for all cases.
  line = start + f'({argstring});'
  return line.rstrip(';')
