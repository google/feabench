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

"""Helper methods related to the evaluation pipelines."""

import collections
import json
import os
from typing import Any, Collection, Sequence

from engmod.common import constants
from engmod.common import file_utils
import numpy as np
import pandas as pd


DOC_DIR = constants.PHY_DOC_DIR
PHY_INTERFACE_PATH = os.path.join(DOC_DIR, 'physics_interfaces.json')


def store_all_physics_interfaces(jsonldir, save_path):
  """Store the set of all physics interfaces used by problems in a directory."""
  physics_interface_set = set({})

  # get paths to all jsons in directory
  files = np.array(file_utils.listDir(jsonldir))
  dfpath = files[np.array([f.endswith('.csv') for f in files])][0]
  df = pd.read_csv(os.path.join(jsonldir, dfpath))
  jsonls = df['jsonl_path'].unique()
  for path in jsonls:
    file_interfaces = []
    with file_utils.Open(path, 'r') as f:
      linelist = list(f)
      print(f'Retrieving {len(linelist)} tasks in {path}')
      promptlist = [json.loads(prompt) for prompt in linelist]
      for prompt in promptlist:
        file_interfaces.extend(prompt['physics_interfaces'])
    physics_interface_set.update(file_interfaces)
  # save
  with file_utils.file_open(save_path, 'w') as f:
    json.dump(list(physics_interface_set), f)
  return physics_interface_set


def convert_string_to_list(answer: str) -> Sequence[str]:
  intlist = answer.strip(']').strip('[').replace("'", '').split(',')
  return [i.strip(' ') for i in intlist]


def remove_items(
    input_args: dict[str, Any], keys_to_ignore: Collection[str]
) -> dict[str, Any]:
  return {k: v for k, v in input_args.items() if k not in keys_to_ignore}


def are_answers_in_options(
    answer: Sequence[str], options: Sequence[str]
) -> bool:
  return all([ans in options for ans in answer])


def have_identical_elements(
    answer: Collection[str], ground_truth: Collection[str]
) -> bool:
  """Returns True if the set of elements match regardless of counts."""
  return set(answer) == set(ground_truth)


def average_correct_with_occurrences(
    answer: Collection[str], ground_truth: Collection[str]
) -> float:
  """Computes average recall/prec per type of interface/feature."""
  if ground_truth:
    gt_elems = collections.Counter(ground_truth)
    answer_elems = collections.Counter(answer)
    avg_recall = 0
    for gt in gt_elems:
      avg_recall += min(
          (answer_elems[gt] / gt_elems[gt]), 1.0
      )  # Answer_freq / GT_freq (recall per feature)
    avg_recall = avg_recall / len(gt_elems)  # average (over all feature TYPES)
    return 100 * avg_recall
  else:
    raise ValueError('Handle zero length targets externally.')


def count_correct_with_occurrences(
    answer: Collection[str], ground_truth: Collection[str]
) -> float:
  """Computes recall/precision per type of interface/feature."""
  if ground_truth:
    gt_elems = collections.Counter(ground_truth)
    answer_elems = collections.Counter(answer)
    count_answer = 0
    count_gt = 0
    for gt in gt_elems:
      print(gt)
      count_gt += gt_elems[gt]
      count_answer += min(answer_elems[gt], gt_elems[gt])
      # recall shouldn't exceed 100%
      print(gt_elems[gt], answer_elems[gt])
    recall = 100 * count_answer / count_gt
    return recall
  else:
    raise ValueError('Handle zero length targets externally.')


def percentage_correct_of_ground_truth(
    answer: Collection[Any], ground_truth: Collection[Any]
) -> float:
  """Returns %age of ground truth elements (not frequency) that exist in answer."""
  if ground_truth:
    count = 0
    for gt in ground_truth:
      if gt in answer:
        count += 1
    return 100 * count / len(ground_truth)
  else:
    raise ValueError('Handle zero length targets externally.')


def percentage_correct_of_ground_truth_dictionaries(
    answer: Collection[dict[str, Any]],
    ground_truth: Collection[dict[str, Any]],
    ignore_keys: Collection[str],
) -> float:
  """Returns metrics for dictionaries."""
  answer_processed = [remove_items(args, ignore_keys) for args in answer]
  gt_processed = [remove_items(args, ignore_keys) for args in ground_truth]
  return percentage_correct_of_ground_truth(answer_processed, gt_processed)
