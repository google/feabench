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

"""Runs an agent experiment on a set of problems."""

import concurrent.futures
import os
from typing import List, Optional

from absl import logging
import ml_collections as mlc
import yaml

from engmod.common import agent_configs
from engmod.common import corrector_main_agents
from engmod.common import file_utils
from engmod.common import simple_agents

AGENTS = {
    'SingleStepAgent': simple_agents.SingleStepAgent,
    'SingleStepMultipleLMAgent': simple_agents.SingleStepMultipleLMAgent,
    'EvolveMainAgent': corrector_main_agents.EvolveMainAgent,
}


def run_experiment(
    experiment_config: mlc.ConfigDict,
    parallelize: bool = False,
) -> List[Optional[agent_configs.GenericAgent]]:
  """Run an experiment that generates code given config."""

  experiment_output_dir = experiment_config.output_base_dir
  file_utils.makedirs(experiment_output_dir)
  file_utils.makedirs(experiment_config.artifact_path)
  # Save the experiment config too for reproducibility.
  # remove any keys incompatible with yaml
  comsol_client = None
  if 'environment' in experiment_config.agent.keys():
    comsol_client = experiment_config.agent.environment.comsol_client
    experiment_config.agent.environment.comsol_client = None
  with file_utils.file_open(
      os.path.join(experiment_output_dir, 'config.yaml'), 'w'
  ) as f:
    yaml.dump(experiment_config.to_yaml(), f, default_flow_style=False)
  if 'environment' in experiment_config.agent.keys():
    experiment_config.agent.environment.comsol_client = comsol_client

  def run_on_problem(problem_name: str) -> Optional[agent_configs.GenericAgent]:
    # Skip if already exists.
    if file_utils.file_exists(
        os.path.join(experiment_output_dir, problem_name)
    ):
      if experiment_config.skip_existing:
        logging.info('Skipping %s already exists.', problem_name)
        return None
      else:
        raise ValueError(f'Problem {problem_name} already exists.')
    else:
      agent = AGENTS[experiment_config.agent.agent_class](experiment_config)
      agent.run_agent(problem_name)
      agent.save_states(out_dir=agent.out_dir)
      return agent

  if parallelize:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
      problem_agents = list(
          executor.map(run_on_problem, experiment_config.dataset.problem_list)
      )
  else:
    problem_agents = []
    for problem_name in experiment_config.dataset.problem_list:
      problem_agents.append(run_on_problem(problem_name))
  return problem_agents
