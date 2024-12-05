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

"""Error analysis and trajectory plotting utilities."""

from collections.abc import Sequence
import json
import re
from typing import Any, Literal

from engmod.common import constants
from engmod.common import file_utils
import matplotlib.pyplot as plt
import numpy as np
import scipy


CODE_PATTERNS = {
    'param': r'model\.param.*?\);',
    'geometry': r'model\.component\("comp1"\)\.geom.*?\);',
    'material': r'model\.component\("comp1"\)\.material.*?\);',
    'physics': r'model\.component\("comp1"\)\.physics.*?\);',
    'mesh': r'model\.component\("comp1"\)\.mesh.*?\);',
    'selection': r'model\.component\("comp1"\)\.selection.*?\);',
    'study': r'model\.study.*?\);',
    'solver': r'model\.sol.*?\);',
    'result': r'model\.result.*?\);',
    'multiphysics': r'model\.component\("comp1"\)\.multiphysics.*?\);',
}

BLOCKS = [
    'param',
    'geometry',
    'physics',
    'material',
    'mesh',
    'solver',
    'result',
]
pattern2block = {
    'param': 'param',
    'geometry': 'geometry',
    'selection': 'geometry',
    'physics': 'physics',
    'multiphysics': 'physics',
    'material': 'material',
    'mesh': 'mesh',
    'solver': 'solver',
    'study': 'solver',
    'result': 'result',
}


def get_stats_for_output(
    elem: tuple[dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
  """Get blockwise error stats for a single state in corrector_main."""
  blockwise = {k: {'total': 0, 'error': 0} for k in BLOCKS}
  for il, line in enumerate(elem[0]['CodeBlock']):
    for pattern in CODE_PATTERNS:
      if re.match(CODE_PATTERNS[pattern], line):
        blockwise[pattern2block[pattern]]['total'] += 1
        if elem[1]['errmask'][il]:
          blockwise[pattern2block[pattern]]['error'] += 1
  return blockwise


def get_best_solution_and_eval_for_problem(
    exp_dir: str, problem: str
) -> tuple[dict[str, Any], str, bool, tuple[dict[str, Any], dict[str, Any]]]:
  """Identifies the best solution for a Multi-Turn experiment problem.

  This is useful to identify in which problems correction resulted in an
  improvement and how many problems it took to get the best solution.

  Args:
    exp_dir: Experiment directory for a Multi-Turn experiment.
    problem: Problem name.

  Returns:
    best_solution_eval: The best solution eval from `evals_best_of_best.json`
    state: Which state in the trajectory does the best solution correspond to.
    Inferred from target path.
    is_init: Whether the best solution is in the initial population sampled.
    best_solution: The best solution. This is a tuple of (LMSolution, Eval). The
      eval in this tuple is the eval done * during * the experiment, i.e.
      agnostic of any Ground truth information.
  """

  best_solution_eval = json.load(
      file_utils.file_open(
          f'{exp_dir}/{problem}/evals/evals_best_of_best.json', 'r'
      )
  )
  target_path = best_solution_eval['lm_target_path']
  target_state = target_path[
      target_path.rindex('output_')
      + len('output_') : target_path.rindex('.txt')
  ]
  if 'init' in target_state:
    ini = target_state.rindex('init_sample_') + len('init_sample_')
    state = f'0.{ini}'
    is_init = True
  else:
    state = target_state
    is_init = False
  corr_main = json.load(
      file_utils.file_open(
          f'{exp_dir}/{problem}/corrector_main_all_states.json', 'r'
      )
  )
  # Eval, State_Id, Was it one of the initial, corrector_main_states_elem
  return best_solution_eval, state, is_init, corr_main[state]


def get_values_for_barchart(
    population: Literal['initial', 'best'],
    exp_dir: str,
    clean_out: bool = False,
) -> Any:
  """Aggregate blockwise executability stats for initial or best solutions."""
  if population == 'initial':
    block_list = []
    for problem in constants.BENCHMARK_PROBLEMS:
      states = json.load(
          file_utils.file_open(
              f'{exp_dir}/{problem}/corrector_main_all_states.json', 'r'
          )
      )
      for ist in range(20):
        blockwise = get_stats_for_output(states[f'0.{ist}'])
        block_list.append(blockwise)

  else:
    if population != 'best':
      raise ValueError(f'Unknown population: {population}')
    block_list = []
    for problem in constants.BENCHMARK_PROBLEMS:
      _, state, _, best_solution = get_best_solution_and_eval_for_problem(
          exp_dir, problem
      )
      print(problem, state)
      blockwise = get_stats_for_output(best_solution)
      block_list.append(blockwise)
  aggregated_blockwise_metrics = {
      k: {'error': [], 'total': [], 'exec': []} for k in BLOCKS
  }
  for elem in block_list:
    for k, v in elem.items():
      aggregated_blockwise_metrics[k]['error'].append(v['error'])
      aggregated_blockwise_metrics[k]['total'].append(v['total'])
      aggregated_blockwise_metrics[k]['exec'].append(
          1.0 - (v['error'] / v['total']) if v['total'] else np.nan
      )
  for cat in aggregated_blockwise_metrics:
    for field in aggregated_blockwise_metrics[cat]:
      aggregated_blockwise_metrics[cat][field] = np.array(
          aggregated_blockwise_metrics[cat][field]
      )
  means_total = {
      k: np.nanmean(v['total']) for k, v in aggregated_blockwise_metrics.items()
  }
  means_err = {
      k: np.nanmean(v['error']) for k, v in aggregated_blockwise_metrics.items()
  }
  means_exec = {
      k: np.nanmean(v['exec']) for k, v in aggregated_blockwise_metrics.items()
  }
  stds_exec = {
      k: np.nanstd(v['exec'], ddof=1)
      for k, v in aggregated_blockwise_metrics.items()
  }
  sem_exec = {
      k: scipy.stats.sem(v['exec'][~np.isnan(v['exec'])])
      for k, v in aggregated_blockwise_metrics.items()
  }
  perc25 = {
      k: np.percentile(v['exec'], 25)
      for k, v in aggregated_blockwise_metrics.items()
  }
  perc75 = {
      k: np.percentile(v['exec'], 75)
      for k, v in aggregated_blockwise_metrics.items()
  }
  if clean_out:
    return {
        'total_lines': means_total,
        'errors': means_err,
        'exec': means_exec,
        'stds': stds_exec,
        'sem': sem_exec,
        'p25': perc25,
        'p75': perc75,
    }
  else:
    return (
        means_total,
        means_err,
        means_exec,
        stds_exec,
        sem_exec,
        perc25,
        perc75,
    )


def make_trajectory_from_dict(
    metrics: dict[str, Any], title: str, colors: Sequence[str]
):
  """Plot trajectories for Multi-Turn experiments."""
  means_gtzero = np.zeros((len(constants.BENCHMARK_PROBLEMS), 19))
  zero_vals = []
  for ip, k in enumerate(constants.BENCHMARK_PROBLEMS):
    zero_vals.extend(metrics[k]['Initial'])
    means_gtzero[ip, :] = metrics[k]['Corr']
  # Means over all
  initial_population = np.mean(zero_vals)
  correction_population = np.mean(means_gtzero, axis=0)
  combined = [initial_population] + list(correction_population)
  xt = np.arange(20)[::2]
  plt.figure()
  for i, k in enumerate(constants.BENCHMARK_PROBLEMS):
    # Add a scatter indicating the spread of the initial population
    plt.scatter(
        [0] * 20, metrics[k]['Initial'], alpha=0.3, color=colors[i], s=1
    )
    # Add a unique line for each problem
    plt.plot(
        np.arange(1, 20),
        metrics[k]['Corr'],
        alpha=0.3,
        color=colors[i],
        fmt='.',
        s=1,
    )
  # Add the mean trend over all problems
  plt.plot(np.arange(20), combined, color='k')
  plt.title(title)
  plt.xticks(xt)
  if 'Errors' in title:
    plt.yticks(np.arange(0, 80, 10))
  plt.xlabel('Solution Iteration')
  plt.show()
