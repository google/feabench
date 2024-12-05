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
"""Physics instruction templates, codelines and patterns."""

import json
import re
from typing import Any, Sequence
import numpy as np


class ActionSequence:
  actions: Sequence[tuple[str | None, dict[str, Any] | None]]
  nl_instructions: Sequence[str | None]
  code: Sequence[str | None]

  def initialize_from_actions(
      self, actions: Sequence[tuple[str | None, dict[str, Any]]]
  ):
    self.actions = actions

  def get_action_sequence_string(self) -> str:
    return str(self.actions)


GEOMETRY_PATTERNS = {
    'create_geometry': (
        r'model.component\("comp1"\).geom\(\).create\("geom1",'
        r' (?P<dimension>.*)\);'
    ),
}

PHYSICS_PATTERNS = {
    'create_interface': (  # pylint:disable=line-too-long
        r'model.component\("comp1"\).physics\(\).create\("(?P<physics_tag>[^"]*)",'
        r' "(?P<name>[^"]*)", "(?P<geometry_tag>[^"]*)"\);'
        # This doesn't handle electrolyte interfaces
    ),
    'create_feature': (
        r'model.component\("comp1"\).physics\("(?P<physics_tag>[^"]*)"\).create\("(?P<feature_tag>[^"]*)",'
        r' "(?P<name>[^"]*)", (?P<dimension>.*)\);'
    ),
    'associate_feature_with_named_selection': r'model.component\("comp1"\).physics\("(?P<physics_tag>[^"]*)"\).feature\("(?P<feature_tag>[^"]*)"\).selection\(\).named\("(?P<selection_name>[^"]*)"\);',
    'associate_interface_with_selection_number': r'model.component\("comp1"\).physics\("(?P<physics_tag>[^"]*)"\).selection\(\).set\((?P<selection_number>.*)\);',  # pylint:disable=line-too-long
    'associate_feature_with_selection_number': r'model.component\("comp1"\).physics\("(?P<physics_tag>[^"]*)"\).feature\("(?P<feature_tag>[^"]*)"\).selection\(\).set\((?P<selection_number>.*)\);',
    'modify_interface_property': (  # pylint: disable=line-too-long
        r'model.component\("comp1"\).physics\("(?P<physics_tag>[^"]*)"\).prop\("(?P<property>[^"]*)"\).set\("(?P<param>[^"]*)",'
        r' (?P<value>.*)\);'
    ),
    'modify_feature_property': (
        r'model.component\("comp1"\).physics\("(?P<physics_tag>[^"]*)"\).feature\("(?P<feature_tag>[^"]*)"\).set\("(?P<param>[^"]*)",'
        r' "(?P<value>[^"]*)"\);'
    ),
}  # pylint: disable=line-too-long


def create_interface(
    physics_tag: str, name: str, geometry_tag: str
) -> tuple[str, str, str]:
  return (
      (
          f'Create an Interface with tag {physics_tag} of type {name} on'
          f' Geometry gtag {geometry_tag}.'
      ),
      f"""model.component("comp1").physics().create("{physics_tag}", "{name}", "{geometry_tag}");""",
      PHYSICS_PATTERNS['create_interface'],
  )


def create_feature(
    physics_tag: str, feature_tag: str, name: str, dimension: str
) -> tuple[str, str, str]:
  return (
      (
          f'Under the interface {physics_tag}, create a feature'
          f' {feature_tag} of type {name} with dimension {dimension}.'
      ),
      f"""model.component("comp1").physics("{physics_tag}").create("{feature_tag}", "{name}", {dimension});""",
      PHYSICS_PATTERNS['create_feature'],
  )


def associate_feature_with_named_selection(
    physics_tag: str, feature_tag: str, selection_name: str
) -> tuple[str, str, str]:
  return (
      (
          f'Under the interface {physics_tag}, associate the feature'
          f' {feature_tag} with selection {selection_name}.'
      ),
      f"""model.component("comp1").physics("{physics_tag}").feature("{feature_tag}").selection().named("{selection_name}");""",
      PHYSICS_PATTERNS['associate_feature_with_named_selection'],
  )


def associate_interface_with_selection_number(
    physics_tag: str, selection_number: str
) -> tuple[
    str, str, str
]:  # Rare, eg: model.component("comp1").physics("rotbm").selection().set(6);
  return (
      (
          f'Associate the interface with tag {physics_tag} with selection'
          f' {selection_number}.'
      ),
      f"""model.component("comp1").physics("{physics_tag}").selection().set({selection_number});""",
      PHYSICS_PATTERNS['associate_interface_with_selection_number'],
  )


def modify_interface_property(
    physics_tag: str, property_name: str, param: str, value: str
) -> tuple[str, str, str]:
  return (
      f"""Under the interface {physics_tag}, and the property {property_name} set the parameter {param} to {value}.""",
      f"""model.component("comp1").physics("{physics_tag}").prop("{property_name}").set("{param}", "{value}");""",
      PHYSICS_PATTERNS['modify_interface_property'],
  )


def modify_feature_property(
    physics_tag: str, feature_tag: str, param: str, value: str
) -> tuple[str, str, str]:
  return (
      f"""Under the interface {physics_tag} and feature {feature_tag}, set the parameter {param} to {value}.""",
      f"""model.component("comp1").physics("{physics_tag}").feature("{feature_tag}").set("{param}", "{value}");""",
      PHYSICS_PATTERNS['modify_feature_property'],
  )


def associate_feature_with_selection_number(
    physics_tag: str, feature_tag: str, selection_number: str
) -> tuple[str, str, str]:
  return (
      (
          f'Under the interface {physics_tag}, associate the feature'
          f' {feature_tag} with selection {selection_number}.'
      ),
      f"""model.component("comp1").physics("{physics_tag}").feature("{feature_tag}").selection().set({selection_number});""",
      PHYSICS_PATTERNS['associate_feature_with_selection_number'],
  )


def parse_code(
    code: Sequence[str],
) -> tuple[Sequence[tuple[str | None, dict[str, Any] | None]], Sequence[bool]]:
  """Parse physics code to extract action and arguments."""
  reverse_map = {}
  for k, v in PHYSICS_PATTERNS.items():
    reverse_map.update({v: k})
  action_map = []
  boolean_map = np.ones(len(code), dtype=bool)
  for i, line in enumerate(code):
    for pattern in reverse_map:
      m = re.fullmatch(pattern, line)
      if m:
        args = m.groupdict()
        action_map.append((reverse_map[pattern], args))
        break
    else:
      action_map.append((None, None))
      boolean_map[i] = False
  return action_map, boolean_map



def convert_jsonaction_sequence_output_to_code(
    result: str,
) -> tuple[
    Sequence[tuple[str | None, dict[str, Any] | None]],
    Sequence[str | None],
    Sequence[str | None],
]:
  """Parses a string of a list of [action, args] tuples from an LLM's output.

  Args:
    result: This should not contain double quotes. This assumes the elements in
      the action sequence has single quotes. i.e. The code currently throws an
      error if the input is a json.dumps(<action_sequence>) instead of
      str(<action_sequence>). Breaks for some 3% problems with "value" args.

  Returns:
    A tuple of three lists [Actions, NL Instructions, Code].
  """
  llm_actions = [
      s.strip('\n').strip(' [')
      for s in result.strip('[').strip(']').split('],')
      if s
  ]
  action_pattern = r"('?)(?P<action>[^\']*)('?), {(?P<action_args>.*)}"
  # '? is to handle cases where the llm doesn't output the quotes on the action
  actions = []
  nl_list = []
  codelines = []
  for i, line in enumerate(llm_actions):
    print(i, line)
    m = re.fullmatch(action_pattern, line)
    if m:
      action = m['action']
      action_args = json.loads(
          str(
              '{'
              + m['action_args']
              .replace('\'"', '"')
              .replace('"\'', '"')
              .replace("'", '"')
              + '}'
          )
      )  # Convert single quotes to double and handle args with double quotes
      actions.append((action, action_args))
      if (
          'property' in action_args
      ):  # temporary fix to handle bug: property is a built-in
        action_args['property_name'] = action_args['property']
        del action_args['property']
      nl, code, _ = globals()[action](**action_args)
      codelines.append(code)
      nl_list.append(nl)
    else:
      actions.append((None, None))
      codelines.append(None)
      nl_list.append(None)
  if len(actions) == 1:
    print('Warning: Code likely not parsed correctly.')
  return actions, nl_list, codelines
