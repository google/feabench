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

# pylint: disable=g-missing-from-attributes
"""Metrics to evaluate action template responses."""

import collections
import dataclasses
from typing import Any, Dict, List, Sequence
from engmod.eval import action_templates
from engmod.eval import evaluation_utils
from engmod.eval import constants
import numpy as np


def parse_action_types(
    actions: Sequence[tuple[str | None, dict[str, Any] | None]]
) -> list[str | None]:
  return [act[0] for act in actions]


def get_type_mask(
    actions: Sequence[str | None], action_types: Sequence[str]
) -> list[int]:
  return [id for (id, act) in enumerate(actions) if act in action_types]


def extract_subtype_values(
    action_type: str,
    action_types: list[str],
    args: list[dict[str, Any]],
    extract_keys: list[str],
) -> dict[str, Any]:
  """Return arguments and other specific values for a specific action type."""
  imask = get_type_mask(action_types, [action_type])
  subtype_args = [args[i] for i in imask]
  # extract values of name, dimension etc
  gt_attr = {'args': subtype_args}
  for k in extract_keys:
    gt_attr[k] = [attrib[k] for attrib in subtype_args]
  return gt_attr


def get_mean_scores(
    scores: List[Dict[str, Any] | None],
    relevant_keys: List[str],
    include_bad: bool = True,
) -> Dict[str, float]:
  """Get mean scores over a list of scores."""
  mean_dict = {}

  for k in relevant_keys:
    all_scores = []
    for score in scores:
      if score:  # was parsed correctly and not a nan
        v = score[k]
        all_scores.append(v)
    all_scores = np.array(all_scores)
    nanmask = np.isnan(all_scores)
    number_nan = np.sum(nanmask)
    scoresum = np.sum(all_scores[~nanmask])
    if include_bad:  # divide by Total - Number_nan
      denom = len(scores) - number_nan
      mean_dict[k] = scoresum / denom
      print(f'{k}: Mean over {denom}')
    else:  # divide by only valid values
      denom = len(all_scores[~nanmask])
      mean_dict[k] = scoresum / denom
      print(f'{k}: Mean over {denom}')
  return mean_dict


# Metrics
@dataclasses.dataclass
class PhysicsMetrics:
  """Keys for a metrics dictionary.

  GT: Ground Truth (Target) Code
  LM: Code being tested. Typically the output of a language model (LM).
  Attributes:
    valid_actions: %age of lines that were parsed correctly.
    strict_action_types_match: [bool] 1 if number of each action type matches
      ground truth
    code_precision: %age of LM code in GT code.
    code_recall: %age of GT code in LM code. Interface Metrics:
      interface_recall: %age of GT interfaces in LM output. NAN if no GT interf.
      interface_precision: %age of LM output interfaces in GT.
      interface_realism: %age of LM output interfaces that are valid interfaces.
        Feature Metrics:
      feature_type_recall: "
      feature_type_precision: " Doesn't factor in cases where the same feature
        must be defined multiple times.
      feature_granular_recall: For each GT feature, compute the recall. Average
        over features.
      feature_granular_precision: ". GT -> LM.
      correct_dimension_features: Of the GT features that are in the LM output,
        does the dimension match the GT dimension?
      feature_code_recall:
      feature_code_precision:
    modify_interface_property_recall:
    modify_interface_property_precision:
  """

  valid_actions: float | None = None
  strict_action_types_match: bool | None = None
  interface_recall: float | None = None
  interface_precision: float | None = None
  interface_realism: float | None = None
  feature_type_recall: float | None = None
  correct_dimension_features: float | None = None
  feature_type_precision: float | None = None
  feature_granular_recall: float | None = None
  feature_granular_precision: float | None = None
  code_precision: float | None = None
  code_recall: float | None = None
  interface_code_precision: float | None = None
  interface_code_recall: float | None = None
  feature_code_precision: float | None = None
  feature_code_recall: float | None = None
  modify_interface_property_precision: float | None = None
  modify_interface_property_recall: float | None = None
  modify_feature_property_precision: float | None = None
  modify_feature_property_recall: float | None = None


@dataclasses.dataclass
class ActionSequenceEvaluator:
  """Returns metrics that evaluate a string of a list of (action, arguments).

  metrics: Name and function to compute metric of list(actions) |
  list(gt_actions)
  Returns:
  Metrics:
  """

  def __init__(self):
    self.metrics = {}

  def evaluate_action_lists(
      self, lm_output: str, target: str, subset: Sequence[str] | None = None
  ) -> dict[str, Any]:
    """Metrics to evaluate action sequence lists.

    Args:
      lm_output: JSON-readable language model action sequence.
      target: JSON-readable target action sequence.
      subset: The subset of metrics we want to compute. If we only care about
        computing physics-interface (feature) related metrics set ['interfaces']
        (['features']). Default: ['interfaces', 'features']

    Returns:
      Dictionary with some subset of PhysicsMetrics computed.
    """
    if not subset:
      # default: compute both interface and feature related metrics
      subset = ['interfaces', 'features']

    # Old Code
    if isinstance(lm_output, str):
      # Extract the physics action sequences, lines of code, action types and
      # args for both the language model (LM) output and the ground truth (GT).
      actions, _, lm_code = (
          action_templates.convert_jsonaction_sequence_output_to_code(lm_output)
      )
      gt_actions, _, gt_code = (
          action_templates.convert_jsonaction_sequence_output_to_code(target)
      )
    else:  # Fold this in here instead of jsonizing and rereading
      actions = lm_output['actions']
      lm_code = lm_output['code']
      gt_actions = target['actions']
      gt_code = target['code']
    print('Number of GT actions', len(gt_actions))
    print('Number of LM actions', len(actions))
    lm_act_types = parse_action_types(actions)
    gt_act_types = parse_action_types(gt_actions)
    lm_args = [act[1] for act in actions]
    gt_args = [act[1] for act in gt_actions]

    # check action list types
    action_counter, gt_action_counter = collections.Counter(
        lm_act_types
    ), collections.Counter(gt_act_types)

    metrics = PhysicsMetrics()
    metrics.valid_actions = 100 * (1.0 - lm_code.count(None) / len(actions))
    metrics.strict_action_types_match = action_counter == gt_action_counter

    # Compute all interface related metrics
    if 'interfaces' in subset:
      lm_attr = extract_subtype_values(
          'create_interface', lm_act_types, lm_args, ['name']
      )
      gt_attr = extract_subtype_values(
          'create_interface', gt_act_types, gt_args, ['name']
      )
      lm_interface_args = lm_attr['args']
      gt_interface_args = gt_attr['args']
      lm_interface_names = lm_attr['name']
      gt_interface_names = gt_attr['name']

      if gt_interface_args:
        # How many GT interfaces exist in LM interfaces?
        # This doesn't account for multiple interfaces of the same type (rare)
        metrics.interface_recall = (
            evaluation_utils.percentage_correct_of_ground_truth(
                lm_interface_names, gt_interface_names
            )
        )
        # How many GT interface-args exist in the list of LM interface-args
        # (except tag). Unless there's a difference in the geometry the
        # interfaces are assigned to this is the same as interface_recall.
        metrics.interface_code_recall = (
            evaluation_utils.percentage_correct_of_ground_truth_dictionaries(
                lm_interface_args,
                gt_interface_args,
                ignore_keys=['physics_tag'],
            )
        )

      else:
        # No GT interfaces doesn't really make sense.
        metrics.interface_recall = np.nan
        metrics.interface_code_recall = np.nan

      if lm_interface_args:
        metrics.interface_precision = (
            evaluation_utils.percentage_correct_of_ground_truth(
                gt_interface_names, lm_interface_names
            )
        )
        metrics.interface_realism = (
            evaluation_utils.percentage_correct_of_ground_truth(
                constants.interface_options, lm_interface_names
            )
        )
        metrics.interface_code_precision = (
            evaluation_utils.percentage_correct_of_ground_truth_dictionaries(
                gt_interface_args,
                lm_interface_args,
                ignore_keys=['physics_tag'],
            )
        )

      else:
        metrics.interface_precision = np.nan
        metrics.interface_code_precision = np.nan
        metrics.interface_realism = np.nan

    # Compute all feature metrics.
    if 'features' in subset:
      lm_attr = extract_subtype_values(
          'create_feature', lm_act_types, lm_args, ['name', 'dimension']
      )
      gt_attr = extract_subtype_values(
          'create_feature', gt_act_types, gt_args, ['name', 'dimension']
      )
      lm_feature_args = lm_attr['args']
      gt_feature_args = gt_attr['args']
      lm_features = lm_attr['name']
      gt_features = gt_attr['name']
      gt_features_dims = gt_attr['dimension']

      if gt_feature_args:
        metrics.feature_granular_recall = (
            evaluation_utils.average_correct_with_occurrences(
                lm_features, gt_features
            )
        )
        metrics.feature_type_recall = (
            evaluation_utils.percentage_correct_of_ground_truth(
                list(set(lm_features)), list(set(gt_features))
            )
        )
        metrics.feature_code_recall = (
            evaluation_utils.percentage_correct_of_ground_truth_dictionaries(
                lm_feature_args, gt_feature_args, ignore_keys=['feature_tag']
            )
        )
      else:  # there are no ground truth features
        metrics.feature_type_recall = 0 if lm_features else 100
        metrics.feature_granular_recall = 0 if lm_features else 100
        metrics.feature_code_recall = 0 if lm_features else 100

      if lm_feature_args:
        # check whether the NUMBER of times each feature is created is the same
        metrics.feature_granular_precision = (
            evaluation_utils.average_correct_with_occurrences(
                gt_features, lm_features
            )
        )
        metrics.feature_type_precision = (
            evaluation_utils.percentage_correct_of_ground_truth(
                list(set(gt_features)), list(set(lm_features))
            )
        )
        metrics.feature_code_precision = (
            evaluation_utils.percentage_correct_of_ground_truth_dictionaries(
                gt_feature_args, lm_feature_args, ignore_keys=['feature_tag']
            )
        )
      else:
        metrics.feature_type_precision = np.nan
        metrics.feature_granular_precision = np.nan
        metrics.feature_code_precision = np.nan
      # Check feature dimensions for correctly created features.
      index_feature_steps = get_type_mask(lm_act_types, ['create_feature'])
      dim_match = []
      print(len(gt_features_dims), len(lm_features))
      for name, dim in zip(gt_features, gt_features_dims):
        if name in lm_features:
          i_ifs = lm_features.index(
              name
          )  # Doesn't handle multiple features well?
          dim_match.append(
              dim == lm_args[index_feature_steps[i_ifs]]['dimension']
          )
      if dim_match:
        metrics.correct_dimension_features = (
            100 * dim_match.count(True) / len(dim_match)
        )
      else:
        metrics.correct_dimension_features = np.nan

    # Compute exact code line matches. These are the strictest metrics.
    if gt_code and any(gt_code):
      metrics.code_recall = evaluation_utils.percentage_correct_of_ground_truth(
          lm_code, gt_code
      )
    else:
      metrics.code_recall = (
          0 if lm_code and any(lm_code) else np.nan
      )  # Clearly the GT code doesn't have any relevant physics action code.
    if lm_code:
      metrics.code_precision = (
          evaluation_utils.percentage_correct_of_ground_truth(gt_code, lm_code)
      )
    else:
      metrics.code_precision = np.nan

    # Compute Interface Property related metrics
    lm_attr = extract_subtype_values(
        'modify_interface_property', lm_act_types, lm_args, []
    )
    gt_attr = extract_subtype_values(
        'modify_interface_property', gt_act_types, gt_args, []
    )
    lm_ip_args = lm_attr['args']
    gt_ip_args = gt_attr['args']

    if gt_ip_args:
      metrics.modify_interface_property_recall = (
          evaluation_utils.percentage_correct_of_ground_truth_dictionaries(
              lm_ip_args, gt_ip_args, ignore_keys=['physics_tag']
          )
      )
    else:
      metrics.modify_interface_property_recall = 0 if lm_ip_args else 100
    if lm_ip_args:
      metrics.modify_interface_property_precision = (
          evaluation_utils.percentage_correct_of_ground_truth_dictionaries(
              gt_ip_args, lm_ip_args, ignore_keys=['physics_tag']
          )
      )
    else:
      metrics.modify_interface_property_precision = np.nan

    # Compute Feature property related metrics
    lm_attr = extract_subtype_values(
        'modify_feature_property', lm_act_types, lm_args, []
    )
    gt_attr = extract_subtype_values(
        'modify_feature_property', gt_act_types, gt_args, []
    )
    lm_ip_args = lm_attr['args']
    gt_ip_args = gt_attr['args']

    if gt_ip_args:
      metrics.modify_feature_property_recall = (
          evaluation_utils.percentage_correct_of_ground_truth_dictionaries(
              lm_ip_args, gt_ip_args, ignore_keys=['physics_tag', 'feature_tag']
          )
      )
    else:
      metrics.modify_feature_property_recall = 0 if lm_ip_args else 100
    if lm_ip_args:
      metrics.modify_feature_property_precision = (
          evaluation_utils.percentage_correct_of_ground_truth_dictionaries(
              gt_ip_args, lm_ip_args, ignore_keys=['physics_tag', 'feature_tag']
          )
      )
    else:
      metrics.modify_feature_property_precision = np.nan

    metrics = metrics.__dict__
    return metrics


def convert_code_to_action_sequence(
    code: str,
    get_split_codelines: bool = False,
) -> (
    tuple[Sequence[Any], Sequence[bool]]
    | tuple[Sequence[Any], Sequence[bool], Sequence[str | None]]
):
  codelines = code.split(';\n')
  codelines = [(c + ';' if not c.endswith(';') else c) for c in codelines]
  action_sequence, action_mask = action_templates.parse_code(codelines)
  # We need the line below since parse_code returns tuples and we need [].
  if get_split_codelines:
    # to match convert json sequence output, only keep lines with a physics
    # action.
    for i, is_phy in enumerate(action_mask):
      if not is_phy:
        codelines[i] = None
    return action_sequence, action_mask, codelines
  else:
    return [[elem[0], elem[1]] for elem in action_sequence], action_mask
