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

"""Analysis utils for leaderboard."""

# pylint: disable=pointless-string-statement

import json
import os
from typing import Any, Optional, Sequence
import warnings

from engmod.common import file_utils
import numpy as np
import pandas as pd
import scipy


def collect_evals(
    eval_dir: str,
    fname: str,
    prob_subset: Optional[list[str]] = None,
    eval_subdir: str = 'evals',
) -> dict[str, Any]:
  """Aggregate already saved evals for an experiment.

  Args:
    eval_dir: Directory containing the evals
    fname: Filename of the eval
    prob_subset: List of problems to include
    eval_subdir: The subdirectory containing the evals

  Returns:
    Aggregated metrics across all problems.
  """
  metrics = {}
  exec_scores = []
  n_err = []
  code_diff = []
  tree_diff = []
  probs = []
  interface_factuality = []
  interface_recall = []
  feature_recall = []
  feature_dim = []
  fprop_recall = []
  exported_tables = []
  target_relerrs = []
  lm_judged_export = []
  print(f'Evaluating directory {eval_dir}')
  fnames = file_utils.listdir(eval_dir)
  for prob in fnames:
    if (
        (prob_subset)
        and (prob in prob_subset)
        and file_utils.isdir(os.path.join(eval_dir, prob))
    ):
      eval_path = os.path.join(eval_dir, prob, eval_subdir, fname)
      if file_utils.file_exists(eval_path):
        # Add evals for the problem.
        probs.append(prob)
        eval_state = json.load(file_utils.file_open(eval_path, 'r'))
        if 'ParsingFailure' in eval_state:
          code_diff.append(0.0)
          interface_factuality.append(0.0)
          interface_recall.append(0.0)
          feature_recall.append(0.0)
          fprop_recall.append(0.0)
          feature_dim.append(0.0)
          exec_scores.append(0.0)
          n_err.append(0.0)
          tree_diff.append(0.0)
          exported_tables.append(False)
          lm_judged_export.append(False)
          target_relerrs.append(np.nan)
        else:
          code_diff.append(float(eval_state['code_diff_score']))
          # Physics Metrics
          if eval_state['physics_metrics']:
            interface_factuality.append(
                float(eval_state['physics_metrics']['interface_realism'])
            )
            interface_recall.append(
                float(eval_state['physics_metrics']['interface_code_recall'])
            )
            feature_recall.append(
                float(eval_state['physics_metrics']['feature_granular_recall'])
            )
            feature_dim.append(
                float(
                    eval_state['physics_metrics']['correct_dimension_features']
                )
            )
            fprop_recall.append(
                float(
                    eval_state['physics_metrics'][
                        'modify_feature_property_recall'
                    ]
                )
            )
          else:
            interface_factuality.append(np.nan)
            interface_recall.append(np.nan)
            feature_recall.append(np.nan)
            fprop_recall.append(np.nan)
            feature_dim.append(np.nan)

          # Executability Metrics
          if 'non_exec_only' in eval_state and eval_state['non_exec_only']:
            exec_scores.append(np.nan)
            n_err.append(np.nan)
            tree_diff.append(np.nan)
            exported_tables.append(False)
            lm_judged_export.append(False)
            target_relerrs.append(np.nan)
          else:
            exec_scores.append(
                float(eval_state['executability_metrics']['Executability'])
            )
            n_err.append(
                int(eval_state['executability_metrics']['Number_of_Errors'])
            )
            tree_diff.append(float(eval_state['tree_diff_score']))
            # Table Metrics
            if eval_state['lm_parsed_output'] != 'None':
              exported_tables.append(
                  bool(eval_state['lm_parsed_output']['table_data'])
              )
              lm_judged_export.append(eval_state['lm_judged_export'] == 'YES')
            else:
              exported_tables.append(False)
              lm_judged_export.append(False)

            # [] gets cast to False
            target_relerr = float(eval_state['target_relative_error'])
            target_relerrs.append(target_relerr)
            if target_relerr < 1:
              print(prob, target_relerr)
      else:
        warnings.warn(f'Eval path {eval_path} does not exist.')
  metrics['Executability'] = np.array(exec_scores)
  metrics['Number_of_Errors'] = np.array(n_err)
  metrics['Code_Diff'] = np.array(code_diff)
  metrics['Tree_Diff'] = np.array(tree_diff)
  metrics['Problems'] = probs
  metrics['Interface_Factuality'] = np.array(interface_factuality)
  metrics['Interface_Recall'] = np.array(interface_recall)
  metrics['Feature_Recall'] = np.array(feature_recall)
  metrics['Feature_Property_Recall'] = np.array(fprop_recall)
  metrics['Exported_Table'] = np.array(exported_tables)
  metrics['Target_Relative_Error'] = np.array(target_relerrs)
  metrics['Feature_Dimension'] = np.array(feature_dim)
  metrics['LM_Judged_Export'] = np.array(lm_judged_export)
  return metrics


def make_leaderboard(
    dirnames: Sequence[str],
    eval_subdirs: Sequence[str],
    eval_filenames: Sequence[str],
    metric_subset: list[str],
    problem_subset: Optional[list[str]] = None,
    exp_descriptions: Optional[str] = None,
    rename_keys: Optional[dict[str, str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Make leaderboard.

  Args:
    dirnames: List of experiment directories.
    eval_subdirs: List of subdirectories containing the evals.
    eval_filenames: The eval filename to use within each
      experiment/problem/evals directory.
    metric_subset: List of metrics to display.
    problem_subset: List of problems to include
    exp_descriptions: List of experiment descriptions
    rename_keys: Dict of old key to new key.

  Returns:
    Dataframes of mean, std, and sem of metrics for each experiment
  Raises:
    ValueError: If the dataframe lengths are inconsistent.
  """
  count_metrics = [
      'Exported_Table',
      '# Target_Relative_Error<1',
      '# LM_Judged_Export',
  ]

  metrics_mean = {met: [] for met in metric_subset + ['Problems']}
  metrics_std = {
      met: []
      for met in metric_subset + ['Problems']
      if met not in count_metrics
  }
  metrics_sem = {
      met: []
      for met in metric_subset + ['Problems']
      if met not in count_metrics
  }
  # std and sem don't make sense if we are counting things

  print('test0')
  for exp_dir, eval_subdir, eval_file in zip(
      dirnames, eval_subdirs, eval_filenames
  ):
    # For each experiment
    if not file_utils.isdir(exp_dir):
      raise ValueError(f'{exp_dir} does not exist.')
    # Aggregate evals for each problem
    combined_evals = collect_evals(
        exp_dir, eval_file, problem_subset, eval_subdir=eval_subdir
    )
    # Take Means, Stds, and SEMs for each metric
    probs = str(combined_evals['Problems'])
    metrics_mean['Problems'].append(probs)
    metrics_std['Problems'].append(probs)
    metrics_sem['Problems'].append(probs)

    # Compute means, stds and sems for most metrics, and do whatever other
    # operations are needed for the rest.
    for met in metric_subset:
      if met == 'Exported_Table':
        metrics_mean['Exported_Table'].append(
            f"{np.sum(combined_evals['Exported_Table'])}/{len(combined_evals['Exported_Table'])}"
        )
      elif met in ['# Target_Relative_Error<1', '# LM_Judged_Export']:
        # Since this is computed below.
        pass
      elif met == 'Relative_Error if <1':
        reasonable_target_mask = combined_evals['Target_Relative_Error'] < 1
        metrics_mean['# Target_Relative_Error<1'].append(
            f'{np.sum(reasonable_target_mask)}/{len(reasonable_target_mask)}'
        )
        if np.any(reasonable_target_mask):
          print(
              f'Exp {exp_dir}. {np.sum(reasonable_target_mask)} exports with'
              ' Target_Relative_Error<1. On Problems:',
              np.array(combined_evals['Problems'])[reasonable_target_mask],
              ' with target relative errors: ',
              combined_evals['Target_Relative_Error'][reasonable_target_mask],
          )
          # Compute stats for only reasonable targets.
          reasonable_targets = combined_evals['Target_Relative_Error'][
              reasonable_target_mask
          ]
          metrics_mean['Relative_Error if <1'].append(reasonable_targets.mean())
          metrics_std['Relative_Error if <1'].append(
              np.std(reasonable_targets, ddof=1)
          )
          metrics_sem['Relative_Error if <1'].append(
              scipy.stats.sem(reasonable_targets)
          )
        else:
          metrics_mean['Relative_Error if <1'].append(np.nan)
          metrics_std['Relative_Error if <1'].append(np.nan)
          metrics_sem['Relative_Error if <1'].append(np.nan)
      elif met == 'Relative_Error|LM_Judged_Export':
        reasonable_target_mask = combined_evals['LM_Judged_Export']
        metrics_mean['# LM_Judged_Export'].append(
            f'{np.sum(reasonable_target_mask)}/{len(reasonable_target_mask)}'
        )
        if np.any(reasonable_target_mask):
          # if any problems had a successful export.
          print(
              f'Exp {exp_dir}. {np.sum(reasonable_target_mask)} exports with'
              ' LM_Judged_Export: True. On Problems:',
              np.array(combined_evals['Problems'])[reasonable_target_mask],
              ' with target relative errors: ',
              combined_evals['Target_Relative_Error'][reasonable_target_mask],
          )
          # Compute stats for only reasonable targets.
          reasonable_targets = combined_evals['Target_Relative_Error'][
              reasonable_target_mask
          ]
          metrics_mean['Relative_Error|LM_Judged_Export'].append(
              reasonable_targets.mean()
          )
          metrics_std['Relative_Error|LM_Judged_Export'].append(
              np.std(reasonable_targets, ddof=1)
          )
          metrics_sem['Relative_Error|LM_Judged_Export'].append(
              scipy.stats.sem(reasonable_targets)
          )
        else:
          metrics_mean['Relative_Error|LM_Judged_Export'].append(np.nan)
          metrics_std['Relative_Error|LM_Judged_Export'].append(np.nan)
          metrics_sem['Relative_Error|LM_Judged_Export'].append(np.nan)
      elif met == 'Relative_Error|Strict':
        reasonable_target_mask = combined_evals['LM_Judged_Export']
        metrics_mean['# LM_Judged_Export'].append(
            f'{np.sum(reasonable_target_mask)}/{len(reasonable_target_mask)}'
        )
        if np.any(reasonable_target_mask):
          # if any problems had a successful export.
          print(
              f'Exp {exp_dir}. {np.sum(reasonable_target_mask)} exports with'
              ' LM_Judged_Export: True. On Problems:',
              np.array(combined_evals['Problems'])[reasonable_target_mask],
              ' with target relative errors: ',
              combined_evals['Target_Relative_Error'][reasonable_target_mask],
          )
          # Compute stats for only reasonable targets AND filter by <5%.
          reasonable_target_mask = combined_evals['LM_Judged_Export'] * (
              combined_evals['Target_Relative_Error'] < 0.1
          )
          reasonable_targets = combined_evals['Target_Relative_Error'][
              reasonable_target_mask
          ]
          metrics_mean['Relative_Error|Strict'].append(
              reasonable_targets.mean()
          )
          metrics_std['Relative_Error|Strict'].append(
              np.std(reasonable_targets, ddof=1)
          )
          metrics_sem['Relative_Error|Strict'].append(
              scipy.stats.sem(reasonable_targets)
          )
        else:
          metrics_mean['Relative_Error|Strict'].append(np.nan)
          metrics_std['Relative_Error|Strict'].append(np.nan)
          metrics_sem['Relative_Error|Strict'].append(np.nan)
      else:
        # For all other metrics, take the mean of all non-nan values.
        npmask = np.isnan(combined_evals[met])
        if np.any(npmask):
          print(f'Skipping {np.isnan(combined_evals[met]).sum()} nans in {met}')

        if np.all(npmask):
          mean, std, sem = np.nan, np.nan, np.nan
        else:
          mean = combined_evals[met][~npmask].mean()
          std = np.std(combined_evals[met][~npmask], ddof=1)
          sem = scipy.stats.sem(combined_evals[met][~npmask])
          print(f'{met}: Mean={mean:.2f} Std={std:.2f} Sem={sem:.2f}')

        metrics_mean[met].append(mean)
        metrics_std[met].append(std)
        metrics_sem[met].append(sem)

  if exp_descriptions:
    common_columns = ['Experiment', 'Eval', 'Description']
    common_values = [dirnames, eval_filenames, exp_descriptions]
  else:
    common_columns = ['Experiment', 'Eval']
    common_values = [dirnames, eval_filenames]

  mean_table, std_table, sem_table = (
      {k: v for k, v in zip(common_columns, common_values)},
      {k: v for k, v in zip(common_columns, common_values)},
      {k: v for k, v in zip(common_columns, common_values)},
  )
  mean_table.update(metrics_mean)
  std_table.update(metrics_std)
  sem_table.update(metrics_sem)

  for it, tab in enumerate([mean_table, std_table, sem_table]):
    lengths = [len(tab[k]) for k in tab]
    if len(set(lengths)) != 1:
      init_len = lengths[0]
      for k in tab:
        if len(tab[k]) != init_len:
          raise ValueError(
              f'Inconsistent lengths for Table {it}, {k}:'
              f' {len(tab[k])}!={init_len}'
          )

  mean_table = pd.DataFrame(mean_table)
  std_table = pd.DataFrame(std_table)
  sem_table = pd.DataFrame(sem_table)

  if rename_keys:
    mean_table.rename(columns=rename_keys, inplace=True)
    std_table.rename(columns=rename_keys, inplace=True)
    sem_table.rename(columns=rename_keys, inplace=True)
  return mean_table, std_table, sem_table


def get_eval_trajectories(
    eval_dir: str, filenames: Sequence[str]
) -> dict[str, Any]:
  """Get eval trajectories for a single problem / experiment."""
  metrics = {}
  executability = []
  n_err = []
  code_diff = []
  tree_diff = []
  fname = []

  for eval_path in filenames:
    if file_utils.file_exists(os.path.join(eval_dir, eval_path)):
      eval_state = json.load(
          file_utils.file_open(os.path.join(eval_dir, eval_path), 'r')
      )
      executability.append(
          float(eval_state['executability_metrics']['Executability'])
      )
      n_err.append(int(eval_state['executability_metrics']['Number_of_Errors']))
      code_diff.append(float(eval_state['code_diff_score']))
      tree_diff.append(float(eval_state['tree_diff_score']))
      fname.append(eval_path)
  metrics['Executability'] = np.array(executability)
  metrics['Number_of_Errors'] = n_err
  metrics['code_diff'] = np.array(code_diff)
  metrics['tree_diff'] = np.array(tree_diff)
  metrics['fname'] = fname
  return metrics
