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

"""Main Corrector Agents."""

# pylint: disable=pointless-string-statement
# pylint: disable=undefined_variable

import json
import os
import random
from typing import Any, Callable, Optional
import warnings

from absl import logging  # pylint: disable=unused-import
import numpy as np

from engmod.common import agent_configs
from engmod.common import constants
from engmod.common import file_utils
from engmod.common import llm_client_builder
from engmod.common import prompt_generation
from engmod.common import simple_agents
from engmod.common.agents import corrector_subagent
from engmod.common.eval import api_score
from engmod.common.eval import evaluate_nafems
from engmod.common.eval import parsing_lm_utils
from engmod.common.eval import utils
from engmod.common.remote_service import mph_comsol_client


CORRECTOR_SUBAGENTS = {
    'CorrectorSubAgentBasic': corrector_subagent.CorrectorSubAgentBasic,
}


class Evaluator:
  """Evaluator for the main corrector agent."""

  def __init__(self, comsol_client: mph_comsol_client.MphComsolClient):
    self.comsol_client = comsol_client

  def evaluate(
      self, current_code: parsing_lm_utils.CodeBlock, name: str
  ) -> dict[str, Any]:
    utils.reinitialize_client(
        self.comsol_client, path=None, model_name=name, previous_code=None
    )
    e_score = api_score.score_code_executability(
        current_code.pythonized_code, self.comsol_client
    )
    return e_score


class HybridEvaluator:
  """Evaluator for the main corrector agent.

  In addition to executability, it also checks if the code can export the target
  value, and asks an LM to verify the code if executability exceeds a certain
  threshold.
  """

  def __init__(
      self,
      comsol_client: mph_comsol_client.MphComsolClient,
      verification_template: str,
      model_call_func: Callable[[str], str],
      executability_threshold: float = 0.0,
  ):
    self.comsol_client = comsol_client
    self.exec_evaluator = Evaluator(comsol_client)
    self.verification_template = file_utils.file_open(
        verification_template, 'r'
    ).read()
    self.model_call_func = model_call_func
    self.executability_threshold = executability_threshold
    self.analytical_guess = ''

  def reset_analytical_guess(self, problem_description: str):
    guess_prompt = """I have given you a problem described below. Please return an analytical estimate for what you believe should be the target value. Think step-by-step. Format your answer as STEPS: (your calculations) and FINAL ANSWER: (the final answer).
PROBLEM: {{problem_description}}
REPLY: """
    self.analytical_guess = self.model_call_func(
        prompt_generation.replace_in_prompt(
            {'{{problem_description}}': problem_description}, guess_prompt
        )
    )

  def evaluate(self, context: dict[str, Any], name: str) -> dict[str, Any]:
    """Evaluates the code in the context."""
    code_block, lm_code_reply, problem_description, target_path = (
        context['solution']['CodeBlock'],
        context['solution']['lm_code_reply'],
        context['problem_description'],
        context['solution']['target_path'],
    )
    self.verification_prompt = prompt_generation.replace_in_prompt(
        {
            '{{problem_description}}': problem_description,
        },
        self.verification_template,
    )

    # Exec Fdbk
    e_score = self.exec_evaluator.evaluate(code_block, name)

    # Only do verification if the code is executable beyond the threshold.
    e_score['Export Successful'] = False
    table_status = """The executability was too low to check the export."""

    if file_utils.file_exists(target_path):  # The file exists, add to the dict
      saved_output = self.comsol_client.get_file_contents(target_path)
      context['solution']['exported_table'] = saved_output
      # Exp: We're modifying the solution dict in place here.

    if e_score['Executability'] > self.executability_threshold:
      if file_utils.file_exists(target_path):  # if the file exists
        saved_output = context['solution']['exported_table']
        e_score['exported_table'] = saved_output
        # Parse as table without doing LM Judging
        parsed_table, _, _ = evaluate_nafems.evaluate_table_output(
            saved_output, 1.0, lm_eval=None
        )
        # Choice: we don't do the LM judged export here for now since we do this
        # below using the analytical guess and because that's done in eval.
        # If the table is not empty.
        if parsed_table.table_data:
          condensed_table = (
              parsed_table.column_headers
              + '\n'
              + '\n'.join(parsed_table.table_data)
          )
          table_status = f"""The code was able to successfully export the following table: {condensed_table}. Assess whether this might be the desired target physical quantity in PROBLEM DESCRIPTION.
          You may compare the table data with the analytical guess you made earlier. ANALYTICAL GUESS: {self.analytical_guess} """
          e_score['Export Successful'] = True
          # Export Successful is True ONLY when executability is above the
          # threshold and the table is not empty.
        else:
          # If the table is empty AND we were above exec threshold.
          table_status = (
              """The code exported an empty table to OUTPUT_PATH/output.txt."""
          )
      else:  # If the file doesn't exist.
        table_status = """The code did not export the target value to OUTPUT_PATH/output.txt."""
      # Call verifier whenever executability is above the threshold.
      verification_reply = self.model_call_func(
          prompt_generation.replace_in_prompt(
              {'{{code}}': lm_code_reply, '{{table_input}}': table_status},
              self.verification_prompt,
          )
      )
      e_score['LM_Verifier'] = verification_reply
    else:
      e_score['LM_Verifier'] = table_status

    return e_score


class EvolveMainAgent(agent_configs.GenericAgent):
  """Main Codegen Controller Agent. Implemented to handle multiple errors.

  SOF: "Survival of the Fittest". We have a separate function that decides which
  state to show it as the one that needs to be corrected.

  This manages smaller agents and the overall flow of the correction process.
  """

  def __init__(self, experiment_config):
    super().__init__(experiment_config)
    lm_client = llm_client_builder.build_lm_client(
        experiment_config.agent.language_model.type,
        experiment_config.agent.language_model.model_url,
        experiment_config.agent.language_model.model_config,
    )
    self.model_call_func = lm_client.query
    self.data_dir = experiment_config.dataset.dataset_dir
    self.out_dir = experiment_config.output_base_dir
    self.artifact_path = experiment_config.artifact_path
    self.save_every = experiment_config.agent.correction.save_every
    if not file_utils.file_exists(self.out_dir):
      raise ValueError(f'{self.out_dir} needs to be created.')
    self.action_history = []
    self.all_states = {}

    # Agents
    self.initial_coder_agent = simple_agents.SingleStepMultipleLMAgent(
        self.experiment_config.initial_codegen_config
    )
    self.corrector_agent = CORRECTOR_SUBAGENTS[
        experiment_config.agent.correction.subagent_class
    ](self.experiment_config, self.model_call_func)

    # Evaluation
    self.evaluator = HybridEvaluator(
        self.experiment_config.agent.environment.comsol_client,
        constants.VERIFY_PROMPT_PATH,
        self.model_call_func,
        executability_threshold=self.experiment_config.agent.correction.executability_threshold,
    )
    # Best replies. Dictionary of best replies: 'index_state':
    # (state, score, fitness)
    self.best_replies = {}
    self.track_num_best_replies = (
        experiment_config.agent.correction.track_num_best_replies
    )  # the dictionary will track min(num_best_replies, num_could_export), i.e.
    # if there will always be at least num_best_replies best replies, but a
    # reply that could
    # successfully export will ALSO be added to the best replies.
    # the number of bad experiences to track
    self.max_plug_best_examples_in_context = 3
    # max number of best solutions to show in context. needed for problems where
    # |best_replies| > track_num_best_replies because several solutions export
    # successfully.
    # max number of bad experiences to show in context
    self.n_bad_experience = experiment_config.agent.correction.n_bad_experience

    # Track the element to be popped
    # This is the index relative to overall iterations (all_states)
    self.index_state_minimum = None
    self.minimum_of_best = None

    self.index_state_maximum = None
    self.maximum_of_best = None
    self.last_index = None  # The key of the Last state in all_states (and in
    # best_replies)

  def compute_fitness_criterion(self, score: dict[str, Any]) -> float:
    """1-2 if export successful, 1 if 100% executable, 0-1 otherwise."""
    return int(score['Export Successful']) + score['Executability']

  def recompute_extrema_of_best(self):
    """Recomputes the minimum and maximum of the best replies.

    This should be called after updating the best_replies dict.
    """
    if not self.best_replies:
      raise ValueError('best_replies is empty.')

    best_indices = list(self.best_replies.keys())
    print('Index Best: ', best_indices)
    old_max = self.maximum_of_best
    self.index_state_minimum = best_indices[0]
    self.index_state_maximum = best_indices[0]

    print(self.index_state_minimum, self.index_state_maximum)
    print(self.best_replies.keys())
    self.minimum_of_best, self.maximum_of_best = (
        self.best_replies[self.index_state_minimum][2],
        self.best_replies[self.index_state_maximum][2],
    )

    for index_state, (_, _, fitness) in self.best_replies.items():
      if fitness < self.minimum_of_best:
        self.minimum_of_best = fitness
        self.index_state_minimum = index_state

      if fitness > self.maximum_of_best:
        self.maximum_of_best = fitness
        self.index_state_maximum = index_state
    if (not old_max) or (self.maximum_of_best > old_max):
      self.action_history.append({
          'Action': 'FoundNewBestSolution',
          'Index': self.index_state_maximum,
          'Fitness': self.maximum_of_best,
      })
    return

  def update_best_replies(
      self,
      index_state: str | int,
      current_state: parsing_lm_utils.LMSolution,
      score: dict[str, Any],
  ) -> None:
    """Updates the best replies dict.

    Args:
      index_state: The index of the current state in all_states.
      current_state: The current state dict.
      score: The score dict from the evaluator.

    Returns:
      None

    Add to best replies if a) the reply exported a non-empty table, b) the
    dict of best_replies is not full, and c) the reply is better than the
    minimum AND minimum couldn't export.
    """
    curr_fitness = self.compute_fitness_criterion(score)
    cond1 = score['Export Successful']
    cond2 = len(self.best_replies) < self.track_num_best_replies
    cond3 = bool(self.minimum_of_best) and (curr_fitness > self.minimum_of_best)
    if cond1 or cond2 or cond3:
      # If we're already tracking B best replies and the minimum best reply
      # couldn't export, delete the minimum best reply.
      c1 = len(self.best_replies) >= self.track_num_best_replies
      c2 = (self.index_state_minimum is not None) and (
          not self.best_replies[self.index_state_minimum][1][
              'Export Successful'
          ]
      )
      if self.index_state_minimum:
        print(
            'Current Minimum of Best: ',
            self.index_state_minimum,
            self.best_replies[self.index_state_minimum][2],
            self.best_replies[self.index_state_minimum][1]['Export Successful'],
        )
      delete_minimum = c1 and c2
      if delete_minimum:
        self.action_history.append({
            'Action': 'DeleteMinimumofBestReplies',
            'Index': self.index_state_minimum,
            'Fitness': self.minimum_of_best,
            'ReasonforDelete': f"MaxSize:{c1}. MinimumCouldn'tExport:{c2}",
        })
        del self.best_replies[self.index_state_minimum]

      self.best_replies.update(
          {index_state: (current_state, score, curr_fitness)}
      )
      print(
          f'Adding new best reply: {index_state} with fitness: {curr_fitness}'
      )
      self.recompute_extrema_of_best()
      if self.index_state_minimum:
        print(
            'New Minimum of Best: ',
            self.index_state_minimum,
            self.best_replies[self.index_state_minimum][2],
            self.best_replies[self.index_state_minimum][1]['Export Successful'],
        )
        print(
            'New Maximum of Best: ',
            self.index_state_maximum,
            self.best_replies[self.index_state_maximum][2],
            self.best_replies[self.index_state_maximum][1]['Export Successful'],
        )
      self.action_history.append({
          'Action': 'UpdatedBestReplies',
          'Index': index_state,
          'Fitness': curr_fitness,
          'ReasonforUpdate': (
              f'ReplyExported:{cond1}. BestRepliesNotFull:{cond2}.'
              f' BetterThanMinimum:{cond3} '
          ),
      })
    else:
      # If the dict of best_replies is full and this is not better than the
      # worst, do nothing.
      pass
    return

  def parse_evaluate_and_log(
      self, eval_context: dict[str, Any], update_best: bool = True
  ) -> tuple[parsing_lm_utils.LMSolution, Optional[dict[str, Any]]]:
    """Parse reply into an LMSolution, evaluate it, and update best replies.

    solution is used if the reply was already parsed into code externally.
    Otherwise we parse here.

    Args:
      eval_context: Must contain the following keys: index_state, name,
        problem_description, (solution OR (current_reply AND target_path).
      update_best: Whether to update the best replies.

    Returns:
      current_state: The current state.
      score: The score of the current state.

    Raises:
      KeyError:
    """
    # 1. Parse if it's not already parsed.
    if 'solution' in eval_context:
      current_state = eval_context['solution']
    else:
      if 'target_path' not in eval_context:
        raise KeyError('target_path is not in eval_context.')
      current_reply = eval_context['current_reply']
      current_state = self.corrector_agent.parser.parse(
          current_reply,
          target_path=eval_context['target_path'],
      )

    # 2. Evaluate the current state if it was parsed successfully.
    name = eval_context['name']
    index_state = eval_context['index_state']

    if current_state['ParsingSuccessful']:
      eval_context['solution'] = current_state
      score = self.evaluator.evaluate(eval_context, name)
      if update_best:
        self.update_best_replies(index_state, current_state, score)
      return current_state, score
    else:
      return current_state, None

  def inject_in_context(
      self,
      iterate_on: str,
  ) -> tuple[
      parsing_lm_utils.LMSolution,
      dict[str, Any],
      list[tuple[dict[str, Any], dict[str, Any]]],
      str,
      list[str | int],
  ]:
    """Decides what states and history to show to the corrector agent.

    Args:
      iterate_on: The iterate_on strategy. Determines what solution is shown to
        the corrector agent as the "current solution".

    Returns:
      best_state: The solution plugged into "CURRENT CODE" in the prompt. This
        is a bit of a misnomer since it might actually be the best state or the
        last state, depending on iterate_on.
      best_score: The score of best state.
      history_states: The history of other solutions to show the agent.
      fdbk_demarcator: The demarcator to use for the feedback.
      tries_in_context: The indices of solutions currently chosen in context.
    """
    best_tries = list(self.best_replies.keys())
    if iterate_on == 'Best':
      best_state, best_score = (
          self.best_replies[self.index_state_maximum][0],
          self.best_replies[self.index_state_maximum][1],
      )
      history_states = [
          (self.best_replies[key][0], self.best_replies[key][1])
          for key in best_tries
          if key != self.index_state_maximum
      ]
      tries_in_context = best_tries
      fdbk_demarcator = "### Best Solution's Execution Feedback {{t}}###\n"
    elif iterate_on == 'Random_Best':
      index_state = random.choice(best_tries)
      best_state, best_score = (
          self.best_replies[index_state][0],
          self.best_replies[index_state][1],
      )
      history_states = [
          (self.best_replies[key][0], self.best_replies[key][1])
          for key in best_tries
          if key != index_state
      ]
      tries_in_context = best_tries
      fdbk_demarcator = (
          "### Random Best Solution's Execution Feedback {{t}}###\n"
      )
    elif iterate_on == 'Last':
      best_state, best_score = (
          self.all_states[self.last_index][0],
          self.all_states[self.last_index][1],
      )
      history_states = [
          (self.best_replies[key][0], self.best_replies[key][1])
          for key in best_tries
          if key != self.last_index
      ]
      tries_in_context = best_tries
      tries_in_context.append(self.last_index)
      fdbk_demarcator = "### Last Solution's Execution Feedback {{t}}###\n"
    else:
      raise NotImplementedError(f'iterate_on: {iterate_on} is not implemented.')
    if len(history_states) > self.max_plug_best_examples_in_context:
      # random subset
      history_states = random.sample(
          history_states, self.max_plug_best_examples_in_context
      )
    return (
        best_state,
        best_score,
        history_states,
        fdbk_demarcator,
        tries_in_context,
    )

  def decide_context(self, problem_description: str) -> dict[str, Any]:
    """Chooses the context for the corrector agent."""

    if self.experiment_config.agent.correction.iterate_on == 'Best':
      # Iterate on the best reply.
      # History includes other "best replies".
      (
          best_state,
          best_score,
          history_states,
          fdbk_demarcator,
          tries_in_context,
      ) = self.inject_in_context(iterate_on='Best')

    elif self.experiment_config.agent.correction.iterate_on == 'Random_Best':
      # Iterate on a random best reply.
      # History includes other "best replies".
      (
          best_state,
          best_score,
          history_states,
          fdbk_demarcator,
          tries_in_context,
      ) = self.inject_in_context(iterate_on='Random_Best')

    elif self.experiment_config.agent.correction.iterate_on == 'Last':
      # This should be the same as "MultipleErrorCorrection" with the addition
      # of the Best and Bad examples.
      (
          best_state,
          best_score,
          history_states,
          fdbk_demarcator,
          tries_in_context,
      ) = self.inject_in_context(iterate_on='Last')

    elif self.experiment_config.agent.correction.iterate_on == 'MCMC-like':
      last_fitness = self.compute_fitness_criterion(
          self.all_states[self.last_index][1]
      )
      best_fitness = self.best_replies[self.index_state_maximum][2]

      if last_fitness < best_fitness:
        # If the last reply is worse than the best,
        # select with random acceptance.
        u = np.random.uniform()
        alpha = last_fitness / best_fitness
        print(f'U={u:.2f} , Alpha={alpha:.2f}')
        if u < alpha:
          # Inject the last reply.
          print('Iterating on last')
          (
              best_state,
              best_score,
              history_states,
              fdbk_demarcator,
              tries_in_context,
          ) = self.inject_in_context(iterate_on='Last')

        else:
          # Inject the best reply to iterate on.
          print('Iterating on best')
          (
              best_state,
              best_score,
              history_states,
              fdbk_demarcator,
              tries_in_context,
          ) = self.inject_in_context(iterate_on='Best')

      elif last_fitness == best_fitness:
        # Work with last_fitness.
        (
            best_state,
            best_score,
            history_states,
            fdbk_demarcator,
            tries_in_context,
        ) = self.inject_in_context(iterate_on='Last')

      else:  # last_fitness > best_fitness
        raise ValueError(
            'Something went wrong if last_fitness is greater than best_fitness.'
        )

    else:
      raise NotImplementedError(
          f'iterate_on: {self.experiment_config.agent.correction.iterate_on} is'
          ' not implemented.'
      )
    # Add `n_bad_experience` recent mediocre proposals.
    mediocre_states = []
    if self.n_bad_experience:
      for i in reversed(self.all_states.keys()):  # Check not all_states-1
        if i in tries_in_context:
          continue
        else:
          mediocre_states.append((self.all_states[i][0], self.all_states[i][1]))
          if len(mediocre_states) == self.n_bad_experience:
            break
    good_examples = prompt_generation.render_correction_history(
        history_states,
        prefix="""Here is the history of the other BEST solutions you proposed and the COMSOL execution replies from the API. `Correct` indicates the line was able to execute. `Error` indicates that the API returned an error message which is provided on the same line and the line did not execute. \n ##### BEST SOLUTIONS HISTORY BEGINS #####\n""",
        suffix="""\n ##### Other BEST SOLUTIONS HISTORY ENDS #####\n""",
        demarcator='\n### Best Solution {{t}}###\n',
        evaluation_mode='Hybrid',
    )
    last_few_bad_examples = ''
    if mediocre_states:
      last_few_bad_examples = prompt_generation.render_correction_history(
          mediocre_states,
          prefix="""Here is the history of some recent BAD solutions you proposed and the COMSOL execution replies from the API. `Correct` indicates the line was able to execute. `Error` indicates that the API returned an error message which is provided on the same line and the line did not execute. DO NOT repeat these solutions. \n ##### BAD SOLUTIONS HISTORY BEGINS #####\n""",
          suffix="""\n ##### BAD SOLUTIONS HISTORY ENDS #####\n""",
          demarcator='\n### Bad Solution {{t}}###\n',
          evaluation_mode='Hybrid',
      )
    return {
        'feedback': prompt_generation.render_correction_history(
            [(best_state, best_score)],
            '',
            '',
            fdbk_demarcator,
            evaluation_mode='Hybrid',
        ),
        'history': good_examples + last_few_bad_examples,
        'problem_description': problem_description,
        'current_code': self.corrector_agent.parser.unparse(
            best_state['CodeBlock'], best_state['target_path']
        ),
    }

  def correct_and_iterate(
      self,
      problem_information: dict[str, Any],
  ) -> parsing_lm_utils.LMSolution:
    if not self.all_states:
      warnings.warn(
          'All states is not empty -- The agent might not have been'
          ' reinitialized.'
      )
    name = problem_information['name']
    problem_description = problem_information['problem_description']

    # Intiialize the states and scores.
    self.all_states = {}
    curr_try = 0
    # target_path = os.path.join(self.artifact_path, f'output_{curr_try}.txt')
    # current_state = {}

    # Start loop.
    while curr_try < self.experiment_config.agent.correction.total_tries:
      if curr_try == 0:
        if self.experiment_config.agent.initialize_from_best:
          # Start with the first N <initial_population_size> states of a
          # previous experiment.
          raise NotImplementedError('Was used for debugging, reimplement.')
        else:
          # Generate multiple initial samples.
          initial_outputs = self.initial_coder_agent.run_agent(name)
        # Score the initial samples.
        for i, output in enumerate(initial_outputs):
          index_state = f'0.{i}'
          eval_context = {
              'name': name,
              'index_state': index_state,
              'problem_description': problem_description,
              'solution': output,
          }
          new_state, new_score = self.parse_evaluate_and_log(eval_context)
          self.action_history.append({
              'Action': 'QueryInitialCodingAgent',
              'Reply': output['lm_code_reply'],
          })
          if new_state['ParsingSuccessful']:
            # This only updates the current state and score if the update was
            # successful.
            new_entry = {k: new_score[k] for k in new_score}  # new_score.copy()
            new_entry.update({
                'Current Block': new_state['CodeBlock'],
                'Action': 'Update&Execute',
                'target_path': new_state['target_path'],
            })
            self.action_history.append(new_entry)
            current_state = new_state
            current_score = new_score
            # Else don't need to update current state and score.
            self.all_states.update(
                {index_state: (current_state, current_score)}
            )
        # Choose the best reply from the initial population.
        current_state = self.best_replies[self.index_state_maximum][0]
        self.last_index = self.index_state_maximum

        # Update for the next iteration.
        curr_try += 1
        target_path = os.path.join(self.artifact_path, f'output_{curr_try}.txt')
      else:
        # Step 1: Correct code using "iterate_on" reply in context with exec
        # history.
        index_state = curr_try  # assuming we have only one sample henceforth
        corr_context = self.decide_context(problem_description)
        current_reply = self.corrector_agent.run_agent(corr_context)

        # Step 2: Evaluate code using current reply.
        eval_context = {
            'current_reply': current_reply,
            'name': name,
            'index_state': index_state,
            'target_path': target_path,
            'problem_description': problem_description,
        }
        new_state, new_score = self.parse_evaluate_and_log(eval_context)
        self.action_history.append({
            'Action': 'QueryCodingSubAgent',
            'Reply': current_reply,
        })
        if new_state['ParsingSuccessful']:
          # This only updates the current state and score if the update was
          # successful.
          new_entry = {k: new_score[k] for k in new_score}  # new_score.copy()
          new_entry.update({
              'Current Block': new_state['CodeBlock'],
              'Action': 'Update&Execute',
              'target_path': target_path,
          })
          self.action_history.append(new_entry)
          current_state = new_state
          current_score = new_score
          # Else don't need to update current state and score.
          self.all_states.update({index_state: (current_state, current_score)})
          self.last_index = index_state
          # make updates for next iteration
          curr_try += 1
          target_path = os.path.join(
              self.artifact_path, f'output_{curr_try}.txt'
          )
          # save
          if self.save_every:
            if (curr_try - 1) % self.save_every == 0:
              self.save_states(self.out_dir)

        else:
          print("""Couldn't parse corrected reply into code.""")
    return current_state

  def run_agent(
      self, problem: str | dict[str, Any]
  ) -> parsing_lm_utils.LMSolution:
    """Runs the agent on a single problem."""
    print('########')
    print('Running EvolveMainAgent on problem: ', problem)
    print('########')

    # Make the subdirectories for the problem.
    self.out_dir = os.path.join(self.experiment_config.output_base_dir, problem)
    file_utils.makedirs(self.out_dir)
    self.artifact_path = os.path.join(
        self.experiment_config.artifact_path, problem
    )
    file_utils.makedirs(self.artifact_path)
    entry = json.load(
        file_utils.file_open(self.data_dir + problem + '.json', 'r')
    )
    prob_desc = prompt_generation.get_problem_description_for_task_version(
        self.experiment_config.version, entry
    )

    # make analytical guess in HybridEvaluator
    self.evaluator.reset_analytical_guess(prob_desc)

    final_state = self.correct_and_iterate(
        {
            'name': problem,
            'problem_description': prob_desc,
        },
    )
    return final_state

  def save_states(self, out_dir: str) -> None:
    """Saves the agent states and other data to the out_dir.

    Args:
      out_dir: The directory to save the states to. This should be called with
        the agent's out_dir.

    Returns:
      None
    """
    if out_dir != self.out_dir:
      raise ValueError("This should be called with the agent's out_dir.")
    jsonized_all_states = {}
    for index in self.all_states:
      state, score = self.all_states[index]
      state = {k: utils.jsonize(v) for k, v in state.items()}
      score = {k: utils.jsonize(v) for k, v in score.items()}
      jsonized_all_states.update({index: (state, score)})
    with file_utils.file_open(
        os.path.join(self.out_dir, 'corrector_main_all_states.json'),
        'w',
    ) as f:
      json.dump(jsonized_all_states, f)

    jsonized_history = []
    for entry in self.action_history:
      entry = {k: utils.jsonize(v) for k, v in entry.items()}
      jsonized_history.append(entry)
    with file_utils.file_open(
        os.path.join(self.out_dir, 'corrector_main_action_history.json'),
        'w',
    ) as f:
      json.dump(jsonized_history, f)

    with file_utils.file_open(
        os.path.join(self.out_dir, 'corrector_main_best_replies.json'),
        'w',
    ) as f:
      jsonized_best_replies = {
          k: {
              'state': {
                  subk: utils.jsonize(subv) for subk, subv in v[0].items()
              },
              'score': {
                  subk: utils.jsonize(subv) for subk, subv in v[1].items()
              },
              'fitness': v[2],
          }
          for k, v in self.best_replies.items()
      }
      json.dump(jsonized_best_replies, f)

    self.corrector_agent.save_states(self.out_dir)
    # Also save the analytical guess.
    with file_utils.file_open(
        os.path.join(self.out_dir, 'analytical_guess.txt'), 'w'
    ) as f:
      f.write(self.evaluator.analytical_guess)
    return
