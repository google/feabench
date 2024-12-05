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

"""Evaluate FEABench."""

from collections.abc import Sequence
import dataclasses
import json
import os
from typing import Any, Optional

from engmod.common import file_utils
from engmod.common import prompt_generation
from engmod.common.eval import api_score
from engmod.common.eval import parsing_lm_utils
from engmod.common.eval import utils
from engmod.common.remote_service import mph_comsol_client
import ml_collections as mlc
import numpy as np


@dataclasses.dataclass
class ParsedOutput:
  """Parsed Output Table.

  Attributes:
    table_metadata: The metadata of the table, i.e. all lines that start with
      `%`.
    table_data: Table data, lines without %.
    column_headers: The last line that begins with `%`.
    last_row: The last row is the one that is expected to contain the target
      value.
    last_value: The last value in the last row, which is expected to be the
      target value.
  """

  table_metadata: Sequence[str]
  table_data: Sequence[str]
  column_headers: str
  last_row: str
  last_value: float


def evaluate_table_output(
    saved_table: str,
    target_value: float,
    lm_eval: Optional[dict[str, Any]] = None,
) -> tuple[ParsedOutput, float, dict[str, Any]]:
  """Checks whether the saved output matches the target output.

  Args:
    saved_table: The saved output table.
    target_value: The desired output value.
    lm_eval: Get an LM to look at the saved output and parse into JSON if output
      is valid. Must contain keys: [model_call_func, eval_prompt_template,
      target_description, model_specifications].

  Returns:
    ParsedOutput: The parsed output table.
    compute_relative_error: The relative error between the saved output and the
      target output if a value was saved, else nan. This uses the
      algorithmically
      parsed last value.
    lm_answer_extraction: The LM answer extraction if the LM could parse the
      output.
      LM_Parsed_Table: Raw LM reply of the parsed value | None
      LM_Exported_Value: Raw LM reply converted to dictionary | None
      LM_Judged_Export: YES or NO
  """
  lines = saved_table.split('\n')

  metadata = []
  data = []
  column_headers = ''

  expected_tags = ['% Model: ', '% Version: ', '% Date: ', '% Table: ']

  index = 0
  for line in lines:
    if line:
      if index < 4:  # % line
        if expected_tags[index] not in line:
          raise ValueError(
              f'Expected {expected_tags[index]}, got {line} at line #{index}'
          )
        metadata.append(line)
      elif index == 4:  # column names
        if not line.startswith('%'):
          raise ValueError(f'Expected column names, got {line}')
        metadata.append(line)
        column_headers = line
      else:  # data is only non-empty if there are lines that don't start with %
        if line.startswith('%'):
          raise ValueError(f'Expected data rows, got {line}')
        data.append(line)
      index += 1

  # Extract Target Value: Algorithmically and using an LM
  last_value = np.nan
  extracted_table = None
  exported_value = None
  # if the LM couldn't parse its output, extracted_table, exported_value and
  # lm_judged_export will be (None, None, 'NO')

  if data:
    # Algorithmic Extraction
    data_value = data[-1].split(' ')[-1]
    if 'i' in data_value:
      last_value = complex(data_value.replace('i', 'j'))
    else:
      last_value = float(data_value)

    try:
      # LM Extraction
      if lm_eval:
        extracted_table = lm_eval['model_call_func'](
            prompt_generation.replace_in_prompt(
                {
                    '{{target_description}}': lm_eval['target_description'],
                    '{{table}}': saved_table,
                    '{{problem_description}}': lm_eval['model_specifications'],
                },
                lm_eval['eval_prompt_template'],
            )
        )  # this is the LM reply
        exported_value = parsing_lm_utils.parse_json_output_to_dict(
            extracted_table
        )  # this is the LM reply parsed into a dict

    except ValueError as e:
      print(e, data)

  # Get a thumbs up or down on the basis of whether the LM can parse the output
  lm_judged_export = 'NO'
  if exported_value:  # not None
    if ('N/A' not in str(exported_value['TARGET VALUE'])) and (
        'NA' not in str(exported_value['TARGET VALUE'])
    ):  # we need str here cause sometimes it's an int
      lm_judged_export = 'YES'
      # the LM could parse the output and it's not N/A

  return (
      ParsedOutput(
          table_metadata=metadata,
          table_data=data,
          column_headers=column_headers,
          last_row=data[-1] if data else '',
          last_value=last_value,
      ),
      utils.compute_relative_error(last_value, target_value),
      {
          'LM_Parsed_Table': extracted_table,
          'LM_Exported_Value': exported_value,
          'LM_Judged_Export': lm_judged_export,
      },
  )


class LeaderboardEvaluator:
  """Evaluator class for both the GT aware and GT independent metrics.

  This only creates and evaluates all problems. Uploading to the leaderboard
  is done separately.
  """

  def __init__(
      self,
      comsol_client: mph_comsol_client.MphComsolClient | None,
      experiment_config: mlc.ConfigDict | dict[str, Any],
      parser: parsing_lm_utils.Parser,
      score_target_dir: Optional[str] = None,
      lm_eval: Optional[dict[str, Any]] = None,
      overwrite: bool = False,
      execution_based: bool = True,
      eval_dir: str = 'evals',
  ):
    """Initializes the evaluator.

    Args:
      comsol_client: The comsol client to use for evaluation.
      experiment_config: The experiment config OR a dictionary containing the
        keys, [dataset_dir, output_base_dir, save_name, agent, save_states_dir].
      parser: The parser to use to parse the LM output.
      score_target_dir: The path during eval might be different than when the
        experiment was run. For example, if we ran the experiment on a VM COMSOL
        connection while the eval is run on a local COMSOL connection.
      lm_eval: For LM parsing of Table -> {"TARGET_VALUE": float,
        "TARGET_UNITS": float}
      overwrite: Whether to overwrite existing evals.
      execution_based: Whether to evaluate execution-based metrics.
      eval_dir: The subdirectory to save the evals in. Usually, 'evals'.

    Returns:
      None
    """
    self.comsol_client = comsol_client
    self.experiment_config = experiment_config
    self.parser = parser
    self.score_target_dir = score_target_dir
    self.lm_eval = lm_eval
    if lm_eval:
      for k in ['model_call_func', 'eval_prompt_template']:
        if k not in self.lm_eval:
          raise ValueError(f'Missing {k} in lm_eval.')
    self.overwrite = overwrite
    self.execution_based = execution_based
    self.eval_dir = eval_dir

  def evaluate(self, problem: str) -> None:
    """This saves the evaluated code(s)."""
    # the output files that need to be read in depend on the save_states code
    # for each agent.
    in_dir = self.experiment_config['dataset']['dataset_dir']
    if self.experiment_config['agent']['agent_class'] == 'Large':
      pinput = json.load(file_utils.file_open(in_dir + problem + '.json', 'r'))
      # GT artifacts
      gt_code = pinput['code']
      parsed_gt = self.parser.parse(gt_code)['CodeBlock']
      gt_artifacts = {
          'CodeBlock': parsed_gt,
      }

    else:
      pinput = json.load(file_utils.file_open(in_dir + problem + '.json', 'r'))
      # GT artifacts
      gt_code = pinput['ground_truth_code']
      target_value = float(pinput['target_value'])
      parsed_gt = self.parser.parse(gt_code)['CodeBlock']
      gt_artifacts = {
          'CodeBlock': parsed_gt,
          'target_tree': pinput['target_tree'],
          'target_value': target_value,
          'target_description': pinput['target_description'],
          'model_specifications': pinput['model_specifications'],
      }

    out_dir = os.path.join(self.experiment_config['output_base_dir'], problem)

    if self.experiment_config['agent']['agent_class'] == 'SingleStepAgent':
      # LM save state outputs
      outfile = os.path.join(
          out_dir, f"{self.experiment_config['save_name']}.json"
      )
      output = json.load(file_utils.file_open(outfile, 'r'))
      lm_code_reply = output['reply']
      target_path = output['target_path']
      lm_parsed = self.parser.parse(lm_code_reply, target_path)['CodeBlock']

      # so it can actually save the output to a real file.
      lm_input = {
          'CodeBlock': lm_parsed,
          'target_path': target_path,
      }
      return self.evaluate_problem_outputs(
          problem,
          gt_artifacts,
          [lm_input],
          names=[self.experiment_config['save_name']],
          overwrite=self.overwrite,
          eval_dir=self.eval_dir,
      )

    elif self.experiment_config['agent']['agent_class'] == 'EvolveMainAgent':
      # evaluate all best replies and save the best of those.
      outfile = os.path.join(out_dir, 'corrector_main_best_replies.json')
      output = json.load(file_utils.file_open(outfile, 'r'))
      # evaluate all best replies and save the best of those.
      lm_answers = []
      best_savenames = []
      for index in output:
        lm_code_reply = output[index]['state']['lm_code_reply']
        if self.score_target_dir:
          target_dir = os.path.join(self.score_target_dir, problem)
          file_utils.makedirs(target_dir)
          target_path = os.path.join(target_dir, f'output_t{index}.txt')
        else:
          target_path = output[index]['state']['target_path']
        lm_parsed = self.parser.parse(lm_code_reply, target_path)['CodeBlock']
        lm_answers.append({
            'CodeBlock': lm_parsed,
            'target_path': target_path,
        })
        best_savenames.append(f'best_t{index}')

      self.evaluate_problem_outputs(
          problem,
          gt_artifacts,
          lm_answers,
          names=best_savenames,
          overwrite=self.overwrite,
          eval_dir=self.eval_dir,
      )
      # Additionally save the best of the best.
      find_and_save_best_score(
          self.experiment_config['output_base_dir'],
          [problem],
          'evals_best_of_best.json',
          criterion=(
              'Executability + ExportedValue + TargetAccuracy|LM_Judged_Export'
          ),
          eval_dir=self.eval_dir,
      )
    elif (
        self.experiment_config['agent']['agent_class'] == 'EvolveMainAgent_ALL'
    ):
      # evaluate ALL replies and save.
      outfile = os.path.join(out_dir, 'corrector_main_all_states.json')
      output = json.load(file_utils.file_open(outfile, 'r'))
      # evaluate all replies and save in directory.
      lm_answers = []
      best_savenames = []
      for index in output:
        lm_code_reply = output[index][0]['lm_code_reply']
        if self.score_target_dir:
          target_dir = os.path.join(self.score_target_dir, problem)
          file_utils.makedirs(target_dir)
          target_path = os.path.join(target_dir, f'output_t{index}.txt')
        else:
          target_path = output[index][0]['target_path']
        lm_parsed = self.parser.parse(lm_code_reply, target_path)['CodeBlock']
        lm_answers.append({
            'CodeBlock': lm_parsed,
            'target_path': target_path,
        })
        best_savenames.append(f'state_{index}')

      self.evaluate_problem_outputs(
          problem,
          gt_artifacts,
          lm_answers,
          names=best_savenames,
          overwrite=self.overwrite,
          eval_dir='evals_trajectories',
      )
    elif self.experiment_config['agent']['agent_class'] == 'Large':
      # LM save state outputs
      if self.execution_based:
        raise NotImplementedError(
            'Large agent not supported with execution based metrics.'
        )
      outfile = os.path.join(
          out_dir, f"{self.experiment_config['save_name']}.json"
      )
      output = json.load(file_utils.file_open(outfile, 'r'))
      lm_code_reply = output['reply']
      try:
        lm_parsed = self.parser.parse(lm_code_reply)['CodeBlock']
      except KeyError:
        print(f'Failed to parse {problem}')
        lm_input = {
            'ParsingFailure': True,
            'CodeBlock': None,
            'target_path': 'output.txt',
        }
        return self.evaluate_problem_outputs(
            problem,
            gt_artifacts,
            [lm_input],
            names=[self.experiment_config['save_name']],
            overwrite=self.overwrite,
            eval_dir=self.eval_dir,
        )
      else:
        # so it can actually save the output to a real file.
        lm_input = {
            'CodeBlock': lm_parsed,
            'target_path': 'output.txt',
        }
        return self.evaluate_problem_outputs(
            problem,
            gt_artifacts,
            [lm_input],
            names=[self.experiment_config['save_name']],
            overwrite=self.overwrite,
            eval_dir=self.eval_dir,
        )

    else:
      raise NotImplementedError(
          'Unsupported agent class:'
          f' {self.experiment_config["agent"]["agent_class"]}'
      )

  def evaluate_problem_outputs(
      self,
      problem: str,
      gt_artifacts: dict[str, Any],
      lm_answers: Sequence[dict[str, Any]],
      names: Optional[Sequence[str]] | None = None,
      overwrite: bool = False,
      eval_dir: str = 'evals',
  ) -> None:
    """Evaluate experiment and save the results in an eval directory.

    Args:
      problem: The problem name.
      gt_artifacts: Ground truth artifacts. Includes CodeBlock target_tree
        target_value or just CodeBlock if not execution based.
      lm_answers: LM outputs. Includes CodeBlock target_path
      names: Optional list of names for the eval outputs corresponding to the
        lm_answers. Saved as evals_{names[i]}.json.
      overwrite: Whether to overwrite existing evals.
      eval_dir: The subdirectory to save the evals in. Usually, 'evals'.

    Returns:
      None
    """
    out_dir = os.path.join(self.experiment_config['output_base_dir'], problem)

    if not file_utils.file_exists(out_dir):
      raise ValueError(f"{out_dir} doesn't exist.")
    eval_dir = os.path.join(out_dir, eval_dir)
    if not file_utils.file_exists(eval_dir):
      file_utils.makedirs(eval_dir)

    gt_code = '\n'.join(gt_artifacts['CodeBlock'].code)

    lm_eval_context = None
    if self.lm_eval:
      lm_eval_context = {k: self.lm_eval[k] for k in self.lm_eval}
      lm_eval_context['target_description'] = gt_artifacts['target_description']
      lm_eval_context['model_specifications'] = gt_artifacts[
          'model_specifications'
      ]

    for i, lm_code_parsed in enumerate(lm_answers):
      save_name = f'evals_{names[i]}.json' if names else f'evals_{i}.json'
      save_path = os.path.join(eval_dir, save_name)
      if file_utils.file_exists(save_path) and not overwrite:
        print(f'Skipping {save_path}')
        continue
      else:
        if 'ParsingFailure' in lm_code_parsed:
          evals = {
              'ParsingFailure': True,
              'executability_metrics': 0.0,
              'code_diff_score': 0.0,
              'physics_metrics': 0.0,
              'tree_diff_score': 0.0,
              'lm_target_path': None,
              'state': i,  # This is a bug, should point to the original state,
              # but you;re fine as long as you don't use this anywhere to
              # identify the best solution index. Use target path instead!!
          }
          evals['lm_parsed_output'] = 'None'
          evals['target_relative_error'] = np.nan
        else:
          # Could it parse the code into a code block?
          lm_code_block = lm_code_parsed['CodeBlock']
          lm_code = '\n'.join(lm_code_block.code)
          lm_target_path = lm_code_parsed['target_path']
          # as strings
          #####################################
          # Metrics: Non-Execution based
          #####################################
          # Physics metrics
          phy_score = api_score.physics_code_metrics(lm_code, gt_code)
          # Code diff
          diff_score = api_score.diff_score(
              lm_code.replace(lm_target_path, 'OUTPUT_PATH/output.txt'),
              gt_code,
          )

          if self.execution_based:
            if not self.comsol_client:
              raise ValueError('Comsol client not initialized.')
            # Get gt artifacts needed
            gt_tree = gt_artifacts['target_tree']
            target_value = gt_artifacts['target_value']

            # Executability metrics (Independent of GT)
            utils.reinitialize_client(
                self.comsol_client,
                path=None,
                model_name='model',
                previous_code=None,
            )
            exec_score = api_score.score_code_executability(
                lm_code_block.pythonized_code, self.comsol_client
            )
            exec_score = {k: utils.jsonize(v) for k, v in exec_score.items()}
            lm_model_tree = self.comsol_client.model_tree()
            # Tree diff
            tree_diff = api_score.model_tree_diff(lm_model_tree, gt_tree)

            evals = {
                'executability_metrics': exec_score,
                'code_diff_score': diff_score,
                'physics_metrics': phy_score,
                'tree_diff_score': tree_diff,
                'lm_target_path': lm_target_path,
                'state': i,
            }

            # Evaluate Outputs
            saved_lm_table = self.comsol_client.get_file_contents(
                lm_target_path
            )
            if 'FileNotFoundError' in saved_lm_table:
              evals['lm_parsed_output'] = 'None'
              evals['target_relative_error'] = np.nan
            else:
              lm_computed_table, answer_error, lm_answer_extraction = (
                  evaluate_table_output(
                      saved_lm_table,
                      target_value,
                      lm_eval_context,
                  )
              )
              evals['lm_parsed_output'] = dataclasses.asdict(lm_computed_table)
              evals['target_relative_error'] = answer_error
              evals['lm_answer_extraction'] = lm_answer_extraction
              evals['lm_judged_export'] = lm_answer_extraction[
                  'LM_Judged_Export'
              ]

          else:
            # No execution based metrics
            exec_score = None
            tree_diff = None
            evals = {
                'executability_metrics': exec_score,
                'code_diff_score': diff_score,
                'physics_metrics': phy_score,
                'tree_diff_score': tree_diff,
                'lm_target_path': lm_target_path,
                'state': i,
            }
            evals['lm_parsed_output'] = None
            evals['target_relative_error'] = None
            evals['lm_answer_extraction'] = None
            evals['lm_judged_export'] = None
            evals['non_exec_only'] = True
        json.dump(evals, file_utils.file_open(save_path, 'w'))
    return


def find_and_save_best_score(
    dirpath: str,
    problems: Sequence[str],
    best_name: str,
    criterion: str,
    eval_dir: str,
) -> None:
  """Reads all saved evals and saves best.

  Reads in each eval for a problem directory for an experiment, identifies the
  single best problem using some criterion and saves into another file.

  Args:
    dirpath: The directory containing the problem directories.
    problems: The list of problem names.
    best_name: The name of the file to save the best score to.
    criterion: The criterion to use to determine the best score.
    eval_dir: The subdirectory containing the evals.

  Returns:
    None
  """
  for problem in problems:
    problem_dir = os.path.join(dirpath, problem)
    problem_eval_dir = os.path.join(problem_dir, eval_dir)
    if not file_utils.file_exists(problem_eval_dir):
      print('No Evals found', problem_eval_dir)
      break
    eval_files = [
        f for f in file_utils.listdir(problem_eval_dir) if f.endswith('.json')
    ]
    print(problem, eval_files)
    scores = []

    for eval_file in eval_files:
      if eval_file == best_name:
        continue
      eval_path = os.path.join(problem_eval_dir, eval_file)
      # print(eval_path)
      score = json.load(file_utils.file_open(eval_path, 'r'))
      scores.append((eval_path, score))
    best_path, best_score = identify_best_score(scores, criterion)
    best_score['eval_path'] = best_path

    best_save = os.path.join(problem_eval_dir, best_name)
    json.dump(best_score, file_utils.file_open(best_save, 'w'))
  return


def compute_fitness(score: dict[str, Any], criterion: str) -> float:  # pylint: disable=g-doc-args
  """Compute fitness score according to some criterion.

  Args:
    score: The score dictionary.
    criterion: The criterion to use to determine the fitness score.

  Returns:
    The fitness score. [0-3]

  Unlike the compute fitness during inference, this IS aware of the ground truth
  """
  if criterion == 'Executability + ExportedValue + TargetAccuracy':
    rel_err_contrib = (
        0  # this just checks that target_relative_error is not nan (Binary)
    )
    accuracy_contrib = 0
    if not np.isnan(score['target_relative_error']):
      rel_err_contrib = int(not np.isnan(score['target_relative_error']))
      if score['target_relative_error'] < 1.0:
        accuracy_contrib = 1.0 - score['target_relative_error']
    return (
        float(score['executability_metrics']['Executability'])
        + rel_err_contrib
        + accuracy_contrib
    )
  elif (
      criterion
      == 'Executability + ExportedValue + TargetAccuracy|LM_Judged_Export'
  ):
    rel_err_contrib = 0
    accuracy_contrib = 0
    if not np.isnan(score['target_relative_error']):
      rel_err_contrib = int(not np.isnan(score['target_relative_error']))
      if (
          score['target_relative_error'] < 1.0
          and score['lm_judged_export'] == 'YES'
      ):
        accuracy_contrib = 1.0 - score['target_relative_error']
    return (
        float(score['executability_metrics']['Executability'])
        + rel_err_contrib
        + accuracy_contrib
    )
  else:
    raise ValueError(f'Unsupported criterion: {criterion}')


def identify_best_score(
    scores: Sequence[tuple[str, dict[str, Any]]],
    criterion: str,
) -> tuple[str, dict[str, Any]]:
  """Identifies the best score according to some criterion.

  Args:
    scores: A list of (eval_path, score) tuples.
    criterion: The criterion to use to determine the best score.

  Returns:
    The best eval path and score.
  """
  if criterion in [
      'Executability + ExportedValue + TargetAccuracy',
      'Executability + ExportedValue + TargetAccuracy|LM_Judged_Export',
  ]:
    best_fitness = 0
    best_path = ''
    best_score = {}
    for path, score in scores:
      fitness = compute_fitness(score, criterion)
      if fitness > best_fitness:
        best_fitness = fitness
        best_path = path
        best_score = score
    return (best_path, best_score)
  else:
    raise ValueError(f'Unsupported criterion: {criterion}')
