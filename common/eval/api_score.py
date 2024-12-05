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
"""Methods for scoring generated COMSOL API code ."""

import difflib
import json
from typing import Any, Sequence

import numpy as np

from engmod.common.eval import utils
from engmod.common.remote_service import mph_comsol_client
from engmod.common.eval import action_eval


EMPTY_TREE = """model
├─ parameters
│  └─ Parameters 1
├─ functions
├─ components
├─ geometries
├─ views
├─ selections
├─ coordinates
├─ variables
├─ couplings
├─ physics
├─ multiphysics
├─ materials
├─ meshes
├─ studies
├─ solutions
├─ batches
├─ datasets
├─ evaluations
├─ tables
├─ plots
└─ exports
"""


def diff_score(generated_code, ground_truth):
  """Score the generated code against the ground truth code using diff.

  Args:
    generated_code: The generated code.
    ground_truth: The ground truth that the generated code should resemble.

  Returns:
    The ratio of difference, 2.0*M / T, where T is the total number of elements
    in both sequences, and M is the number of matches. This is 1.0 if the
    sequences are identical, and 0.0 if they have nothing in common.
  """
  matcher = difflib.SequenceMatcher()
  matcher.set_seqs(generated_code, ground_truth)
  return matcher.ratio()


def model_tree_diff(lm_model_tree: str, target_model_tree: str) -> float:
  """Compute diff between model trees of the generated code & ground truth.

  Args:
    lm_model_tree: The model tree of the generated code.
    target_model_tree: The model tree of the ground truth.

  Returns:
    Normalizes the diff score between the score of an empty tree and that of the
    ground truth tree.
  """
  lm_score = diff_score(lm_model_tree, target_model_tree)
  empty_score = diff_score(EMPTY_TREE, target_model_tree)
  return (lm_score - empty_score) / (1.0 - empty_score)


def score_code_executability(
    generated_code: Sequence[str],
    client: mph_comsol_client.MphComsolClient,
) -> dict[str, Any]:
  """Returns lines that run without error/total lines.

  Args:
    generated_code: ALREADY pythonized code lines.
    client: The client to use to run the code. This should have been
      reinitialized externally, i.e. if any previous code should have been run
      on the client, it should have been run before this function is called.

  Returns:
    A dictionary with
      flags: The flags for each line of code.
      replies: The replies from the client for each line of code.
      errmask: A boolean mask indicating which lines of code had errors.
      Number_of_Errors: The number of lines of code that had errors.
      Total Lines: The total number of lines of code.
      Error_Rate: The error rate, Number_of_Errors/Total Lines. Ideal: 0.0
      "Executability": 1.0 - Error_Rate. Ideal: 1.0
  """
  messages, flags = utils.test_code_in_client(client, generated_code)
  flags = np.array(flags)
  errmask = np.array(flags) != 'Correct'
  score = {
      'flags': flags,
      'replies': messages,
      'errmask': errmask,
      'Number_of_Errors': errmask.sum(),
      'Total Lines': len(errmask),
  }
  score['Error_Rate'] = score['Number_of_Errors'] / score['Total Lines']
  score['Executability'] = 1.0 - score['Error_Rate']
  return score


# http://b/328265381 Parse physics code from full code and run special
# physics-only metrics OR check if this breaks on adding irrelevant lines.
def physics_code_metrics(
    generated_code: str, ground_truth: str
) -> dict[str, float] | None:
  """Score generated code against the ground truth code using physics metrics.

  When a certain code type is not present in the ground_truth, the metric is
  usually 100 or nan (for interface_realism, code_recall, interface_code_recall,
  correct_dimension_features).

  Args:
    generated_code: the generated code.
    ground_truth: the ground truth that the generated code should match.

  Returns:
    A dictionary with
      interface_realism: The percentage of generated_code physics interfaces
      that exist.
      ---
      interface_code_recall: The percentage of ground_truth physics interface
      creation calls that exist in the LM code, modulo physics_tag.
      ---
      feature_code_recall: The percentage of ground_truth physics feature
      creation calls that exist in the LM code, modulo feature_tag. Doesn't
      factor in cases where the same feature must be defined multiple times.
      correct_dimension_features: Of the ground_truth features that are also
      in generated_code, does the dimension match the ground_truth dimension?
      ---
      feature_granular_recall: For each ground_truth feature, compute the
      recall. Average over ground_truth features. This DOES penalize the model
      for not creating a feature multiple times, in cases where it should have.
      ---
      modify_interface_property_recall: The percentage of ground_truth physics
      feature creation calls that exist in generated_code, modulo physics_tag.
      ---
      modify_feature_property_recall: The percentage of ground_truth physics
      feature creation calls that exist in generated_code, modulo
      physics_tag+feature_tag.
      ---
      code_recall: How many lines of ground_truth code are in generated_code?
      This is the strictest metric.
  """
  physics_metrics_subset = [  # See action_eval.PhysicsMetrics for the meanings
      'code_recall',
      'interface_realism',
      'interface_code_recall',
      'feature_code_recall',
      'correct_dimension_features',
      'feature_granular_recall',
      'modify_interface_property_recall',
      'modify_feature_property_recall',
  ]
  try:
    processed_answer, _, lm_action_codelines = (
        action_eval.convert_code_to_action_sequence(
            generated_code, get_split_codelines=True
        )
    )
    target_answer, _, gt_action_codelines = (
        action_eval.convert_code_to_action_sequence(
            ground_truth, get_split_codelines=True
        )
    )

    evaluator = action_eval.ActionSequenceEvaluator()
    score = evaluator.evaluate_action_lists(
        {'actions': processed_answer, 'code': lm_action_codelines},
        {'actions': target_answer, 'code': gt_action_codelines},
    )
    subset_score = {
        k: v for k, v in score.items() if k in physics_metrics_subset
    }
    return subset_score
  except json.JSONDecodeError:
    # Some lines of code (esp property related lines) include
    # 'new String[]{"0", "L", "0"}' which breaks this
    # code since it can't be json-ized. We should probably add this to a bug,
    # but it's not * too * common so far.
    return None
