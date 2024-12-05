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

"""Simple LM Agents."""

import concurrent.futures
import json
import os
import time
from typing import Any, List
import numpy as np
from engmod.common import agent_configs
from engmod.common import file_utils
from engmod.common import llm_client_builder
from engmod.common import prompt_generation
from engmod.common.eval import parsing_lm_utils
from engmod.common.eval import utils


class SingleStepAgent(agent_configs.GenericAgent):
  """Single Step Codegen Agent."""

  def __init__(self, experiment_config):
    super().__init__(experiment_config)
    lm_client = llm_client_builder.build_lm_client(
        experiment_config.agent.language_model.type,
        experiment_config.agent.language_model.model_url,
        experiment_config.agent.language_model.model_config,
    )
    self.model_call_func = lm_client.query
    self.data_dir = experiment_config.dataset.dataset_dir
    self.save_name = experiment_config.save_name
    self.out_dir = ''  # this is output_base_dir/problem. Not set yet.
    assert file_utils.file_exists(self.experiment_config.output_base_dir)

  def run_agent(self, problem: str | dict[str, Any]) -> dict[str, Any]:
    """Runs the agent on a single problem."""
    # Prob specific directories.
    self.out_dir = os.path.join(self.experiment_config.output_base_dir, problem)
    self.artifact_path = os.path.join(
        self.experiment_config.artifact_path, problem
    )
    file_utils.makedirs(self.out_dir)
    file_utils.makedirs(self.artifact_path)
    # Get data and replies.
    entry = json.load(
        file_utils.file_open(self.data_dir + problem + '.json', 'r')
    )
    tailored_prompt = prompt_generation.specify_prompt_template(
        self.experiment_config.version,
        self.experiment_config.prompting_strategy.template,
        entry,
    )
    reply = self.model_call_func(tailored_prompt)
    out = {'prompt': tailored_prompt, 'reply': reply}
    # Not parsing here. Parse during eval, since you anyway can't save the class
    target_path = os.path.join(self.artifact_path, f'{self.save_name}.txt')
    out['target_path'] = target_path
    self.out = out
    return out

  def save_states(self, out_dir: str) -> None:
    if not out_dir:
      raise ValueError('Output directory not set. Run agent first.')
    json.dump(
        self.out,
        file_utils.file_open(
            os.path.join(self.out_dir, f'{self.save_name}.json'), 'w'
        ),
    )


class SingleStepMultipleLMAgent(agent_configs.GenericAgent):
  """Single Step Codegen Agent, but with multiple LMs and prompts for each."""

  def __init__(self, experiment_config):
    super().__init__(experiment_config)
    self.lm_clients = []
    for (
        language_model_type,
        language_model_model_url,
        language_model_config,
    ) in zip(
        experiment_config.agent.language_model.types,
        experiment_config.agent.language_model.model_urls,
        experiment_config.agent.language_model.model_configs,
    ):
      lm_client = llm_client_builder.build_lm_client(
          language_model_type,
          language_model_model_url,
          language_model_config,
      )
      self.lm_clients.append(lm_client)
    self.parser = parsing_lm_utils.CodeParser(
        parsing_lm_utils.postprocess_result
    )
    # print(type(self.lm_clients[0]))

    self.data_dir = experiment_config.dataset.dataset_dir
    self.out_dir = experiment_config.output_base_dir
    if not file_utils.file_exists(self.out_dir):
      raise ValueError(f"{self.out_dir} doesn't exist.")
    self.output = None

  def run_agent(
      self, problem: str | dict[str, Any]
  ) -> List[parsing_lm_utils.LMSolution]:
    """Runs the agentS on a single problem."""
    self.out_dir = os.path.join(self.experiment_config.output_base_dir, problem)
    self.artifact_path = os.path.join(
        self.experiment_config.artifact_path, problem
    )
    file_utils.makedirs(self.out_dir)
    entry = json.load(
        file_utils.file_open(
            os.path.join(self.data_dir, (problem + '.json')), 'r'
        )
    )

    def single_query(
        template, client, sname, seed
    ) -> parsing_lm_utils.LMSolution:
      tailored_prompt = prompt_generation.specify_prompt_template(
          self.experiment_config.version,
          template,
          entry,
      )
      print(f'Seed={seed}', flush=True)
      t1 = time.time()
      reply = client.query(tailored_prompt, seed=seed)
      print(f'Reply recvd: {time.time() - t1}', flush=True)

      target_path = os.path.join(self.artifact_path, f'output_{sname}.txt')
      lm_solution = self.parser.parse(reply, target_path)
      print(lm_solution['ParsingSuccessful'], flush=True)
      lm_solution['prompt'] = tailored_prompt
      return lm_solution

    rng = np.random.default_rng(seed=42)
    seeds = rng.choice(
        1000,
        size=len(self.experiment_config.prompting_strategy.templates),
        replace=False,
    )
    if self.experiment_config.agent.WORKER_COUNT > 1:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=self.experiment_config.agent.WORKER_COUNT
      ) as executor:
        outputs = list(
            executor.map(
                single_query,
                self.experiment_config.prompting_strategy.templates,
                self.lm_clients,
                self.experiment_config.save_names,
                seeds,
            )
        )
    else:
      outputs = []
      for (
          template,
          client,
          sname,
          seed,
      ) in zip(
          self.experiment_config.prompting_strategy.templates,
          self.lm_clients,
          self.experiment_config.save_names,
          seeds,
      ):
        outputs.append(single_query(template, client, sname, seed))

    self.outputs = outputs
    return self.outputs

  def save_states(self, out_dir: str) -> None:
    if out_dir != self.out_dir:
      raise ValueError(
          f'out_dir: {out_dir} != self.out_dir: {self.out_dir}. This is not'
          ' supported.'
      )
    for output, save_name in zip(
        self.outputs, self.experiment_config.save_names
    ):
      json.dump(
          {k: utils.jsonize(v) for k, v in output.items()},
          file_utils.file_open(
              os.path.join(out_dir, f'lm_{save_name}_output.json'), 'w'
          ),
      )
