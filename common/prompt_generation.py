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

"""Utils related to generating prompts."""

from typing import Any


def replace_in_prompt(
    placeholder_value_mapping: dict[str, str], prompt: str
) -> str:
  """Safe replace placeholders in prompt.

  Args:
    placeholder_value_mapping: Mapping of placeholder to value.
    prompt: Prompt to infill.

  Returns:
    prompt: Customized prompt.
  Raises:
    ValueError: If placeholder not found in prompt.
  """
  for placeholder, entry_field in placeholder_value_mapping.items():
    if placeholder in prompt:
      prompt = prompt.replace(placeholder, entry_field)
    else:
      raise ValueError(placeholder)
  return prompt


def get_problem_description_for_task_version(
    version: int, entry: dict[str, Any]
) -> str:
  """Get problem description for task version."""
  if version == 0:
    return (
        entry['model_specifications']
        + '\n**SELECTION IDENTITIES:**\n'
        + entry['selection_information']
    )
  elif version == 1:
    return entry['plan']
  else:
    raise NotImplementedError(f'Unsupported experiment version: {version}')


def specify_prompt_template(
    version: int, prompt_template: str, entry: dict[str, Any]
) -> str:
  """Specify prompt template depending on the task version.

  Args:
    version: What version of the experiment. 0 or 1.
      0: ModelSpecs2Code.
      1: Plan2Code.
    prompt_template: Prompt template.
    entry: Problem Entry.

  Returns:
    prompt: Customized prompt.
  """
  mapping = {
      '{{problem_description}}': get_problem_description_for_task_version(
          version, entry
      ),
      '{{target_description}}': entry['target_description'],
  }
  return replace_in_prompt(mapping, prompt_template)


def render_correction_history(
    code_states: list[tuple[dict[str, Any], dict[str, Any]]],
    prefix: str = '',
    suffix: str = '',
    demarcator: str = '\n### Try {{t}}###\n',
    evaluation_mode: str = 'Execution',
) -> str:
  """Renders past code history and feedback.

  Args:
    code_states: List of (state, score) tuples. state has: 'CodeBlock':
      CodeBlock object. 'target_path': Path to the target file that CodeBlock
      saved to. score has: 'replies': List of strings. 'flags': List of strings.
      'LM_Verifier': if evaluation_mode == 'Hybrid', then this is the verifier
      feedback. 'target_path': Path to the target file.
    prefix: Prefix to add to the rendered history.
    suffix: Suffix to add to the rendered history.
    demarcator: Demarcator to add between each try.
    evaluation_mode: Evaluation mode.

  Returns:
    A string containing the correction history.
  """
  if code_states:
    for ist, state in enumerate(code_states):
      # For each state, render the feedback of the evaluator.
      prefix += (
          replace_in_prompt({'{{t}}': str(ist)}, demarcator)
          if '{{t}}' in demarcator
          else demarcator
      )
      code, replies, flags = (
          state[0]['CodeBlock'].code,
          state[1]['replies'],
          state[1]['flags'],
      )
      target_path = state[0]['target_path']
      for line, reply, flag in zip(code, replies, flags):
        # We want the LM to only see the dummy path.
        if target_path in line:
          line = line.replace(target_path, 'OUTPUT_PATH/output.txt')
        if flag == 'Correct':
          msg = f'{line} -> {flag}'
        else:
          msg = f'{line} -> Error: {reply}'
        prefix += f'{msg}\n'
      if evaluation_mode == 'Hybrid':
        verifier_msg = """\nVerifier Feedback: """ + state[1]['LM_Verifier']
        prefix += verifier_msg + '\n'
    return prefix + suffix
  else:
    return 'No Correction History yet'
