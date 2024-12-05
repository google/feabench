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
"""Agent configs for different experiment versions."""

import abc
import datetime
import os
from typing import Any, List, Literal, Optional, Sequence

from engmod.common import constants
from engmod.common import file_utils
from engmod.common import llm_client_builder
from engmod.common.remote_service import mph_comsol_client
import ml_collections as mlc


class GenericAgent(abc.ABC):
  """Abstract base class for agents."""
  out_dir: str

  def __init__(self, experiment_config: mlc.ConfigDict):
    self.experiment_config = experiment_config

  @abc.abstractmethod
  def run_agent(
      self, problem: str | dict[str, Any]
  ) -> dict[str, Any] | str | List[dict[str, Any]]:
    pass

  def save_states(self, out_dir: str) -> None:
    pass


def get_singlestep_config(
    version: int,
    template_path: str,
    agent_class: str = 'SingleStepAgent',
    run: str | None = None,
    model_type: str = 'gemini_external',
    model_url: str = 'gemini-pro-1.5-001',
    model_config: dict[str, Any] | None = None,
    dataset_dir: str = constants.DATA_DIR,
    problem_list: Sequence[str] = constants.BENCHMARK_PROBLEMS,
    artifact_path: str | None = None,
    skip_existing: bool = True,
    name: str = 'baseline',
    save_name: str = 'output',
    exp_directory: str = constants.WORKING_EXPERIMENT_DIR,
) -> mlc.ConfigDict:
  """Codegen Config for ModelSpecs2Code and Plan2Code.

  Args:
    version: What version of the experiment. 0 or 1.
      0: ModelSpecs2Code.
      1: Plan2Code.
    template_path: Path to the prompt template.
    agent_class: Agent class.
    run: Date this was run on. To handle updates to the base dataset.
    model_type: Language Model type, to build LM client and get default config.
    model_url: Language Model URL.
    model_config: Language Model config.
    dataset_dir: Dataset directory.
    problem_list: List of problem names.
    artifact_path: Artifact directory.
    skip_existing: If True, skip problems that already have a solution. If
      False, raise error.
    name: Name of the experiment. A Descriptor eg: "doc0"
    save_name: Name of the output file and artifact file.
    exp_directory: Results are saved to output_base_dir =
      exp_directory/v{version}_{name}/{run}/

  Returns:
    ConfigDict for the experiment.
  """

  config = mlc.ConfigDict()

  # Version
  config.version = version
  if version not in [0, 1]:
    raise ValueError(f'Unsupported experiment version: {version}')

  # Agent specs.
  config.agent = mlc.ConfigDict()
  config.agent.agent_class = agent_class
  config.agent.language_model = mlc.ConfigDict()
  config.agent.language_model.type = model_type
  config.agent.language_model.model_url = model_url
  config.agent.language_model.model_config = model_config

  # Dataset specs.
  config.dataset = mlc.ConfigDict()
  config.dataset.dataset_dir = dataset_dir
  if problem_list is None:
    problem_list = [
        f[: f.rindex('.json')]
        for f in file_utils.listdir(dataset_dir)
        if (f.endswith('json') and f != 'le1.json')
    ]
  config.dataset.problem_list = problem_list

  # LLM Chain specs.
  config.prompting_strategy = mlc.ConfigDict()
  if f'v{config.version}' not in template_path:
    raise ValueError('Mismatch between prompt version and task version.')
  config.prompting_strategy.template_path = template_path
  config.prompting_strategy.template = file_utils.file_open(
      config.prompting_strategy.template_path, 'r'
  ).read()

  # Output location.
  config.run = (
      datetime.datetime.now().strftime('%m-%d_%H-%M') if run is None else run
  )
  config.name = name
  config.output_base_dir = (
      f'{exp_directory}/v{version}_{config.name}/{config.run}/'
  )
  if artifact_path:
    config.artifact_path = os.path.join(
        artifact_path, f'v{version}_{config.name}/{config.run}/'
    )
  else:
    config.artifact_path = None

  config.skip_existing = skip_existing
  config.save_name = save_name
  return config


def get_multiple_lm_config(
    version: int,
    agent_class: Literal['SingleStepMultipleLMAgent'],
    model_types: Sequence[str],
    template_paths: Sequence[str],
    model_urls: Sequence[str],
    model_configs: Sequence[dict[str, Any]],
    save_names: Sequence[str],
    artifact_path: str,
    run: str | None = None,
    dataset_dir: str = constants.DATA_DIR,
    problem_list: Sequence[str] | None = None,
    skip_existing: bool = True,
    name: str = 'multiple_lm',
    workers: int = 4,
    exp_directory: str = constants.WORKING_EXPERIMENT_DIR,
) -> mlc.ConfigDict:
  """Codegen Config for ModelSpecs2Code and Plan2Code.

  Args:
    version: What version of the experiment. 0 or 1.
      0: ModelSpecs2Code.
      1: Plan2Code.
    agent_class: Agent class.
    model_types: Language Model types.
    template_paths: Paths to the prompt templates.
    model_urls: Language Model URLs.
    model_configs: LanguageModel configs.
    save_names: Names to save the output corresponding to each lm_client.
    artifact_path: Artifact directory.
    run: Date this was run on. To handle updates to the base dataset.
    dataset_dir: Dataset directory.
    problem_list: List of problem names.
    skip_existing: If True, skip problems that already have a solution. If
      False, raise error.
    name: Name of the experiment. A Descriptor eg: "doc0"
    workers: Workers for Concurrent.Futures. If 1, simple loop.
    exp_directory: Results are saved to output_base_dir =
      exp_directory/v{version}_{name}/{run}/

  Returns:
    ConfigDict for the experiment.
  """

  config = mlc.ConfigDict()

  # Version
  config.version = version
  if version not in [0, 1]:
    raise ValueError(f'Unsupported experiment version: {version}')

  # Agent specs.
  config.agent = mlc.ConfigDict()
  config.agent.agent_class = agent_class
  config.agent.language_model = mlc.ConfigDict()
  config.agent.language_model.types = model_types
  config.agent.language_model.model_urls = model_urls
  config.agent.language_model.model_configs = model_configs

  # Dataset specs.
  config.dataset = mlc.ConfigDict()
  config.dataset.dataset_dir = dataset_dir
  if problem_list is None:
    problem_list = [
        f[: f.rindex('.json')]
        for f in file_utils.listdir(dataset_dir)
        if (f.endswith('json') and f != 'le1.json')
    ]
  config.dataset.problem_list = problem_list

  # LLM Chain specs.
  config.prompting_strategy = mlc.ConfigDict()
  config.prompting_strategy.template_paths = template_paths
  config.prompting_strategy.templates = [
      file_utils.file_open(tpath, 'r').read() for tpath in template_paths
  ]

  # Output location.
  config.run = (
      datetime.datetime.now().strftime('%m-%d_%H-%M') if run is None else run
  )
  config.name = name
  config.output_base_dir = (
      f'{exp_directory}/v{version}_{config.name}/{config.run}/'
  )
  config.save_names = save_names
  config.artifact_path = os.path.join(
      artifact_path, f'v{version}_{config.name}/{config.run}/'
  )
  # Other
  config.skip_existing = skip_existing  # If True, skip existing problems.
  config.agent.WORKER_COUNT = workers
  return config


def get_evolver_agent_experiment_config(
    codegen_config_args: dict[str, Any],
    corrector_subagent_class: Literal['CorrectorSubAgentBasic'],
    comsol_client: mph_comsol_client.MphComsolClient,
    corrector_prompt_path: str,
    total_tries: int = 10,
    tools: Optional[list[str]] = None,
    tool_selector_prompt_path: Optional[str] = None,
    lib_prompt_path: Optional[str] = None,
    library_path: Optional[str] = None,
    exclude_model_ids_from_library: Optional[int] = None,
    save_every: Optional[int] = None,
    executability_threshold: float = 0.0,
    track_num_best_replies: int = 3,
    n_bad_experience: int = 0,
    iterate_on: Literal['best', 'random_best'] = 'best',
    initial_population_size: int = 1,
    initialize_from_best: Optional[str] = None,
    workers: int = 4,
) -> mlc.ConfigDict:
  """Codegen Corrector Experiment Config for ModelSpecs2Code and Plan2Code.

  Args:
    codegen_config_args: Codegen config args. Args for the "Initial Codegen"
      agent.
    corrector_subagent_class: Corrector subagent class.
    comsol_client: COMSOL client.
    corrector_prompt_path: Path to the corrector prompt.
    total_tries: Total number of tries.
    tools: Names of tools to use.
    tool_selector_prompt_path: Path to the tool selector prompt.
    lib_prompt_path: Path to the RetrieveLibrarySnippets prompt.
    library_path: Path to the Concept Library, if already built and saved.
    exclude_model_ids_from_library: List of model ids to exclude from the
      library.
    save_every: Save states every N tries.
    executability_threshold: At what threshold of executability to run the
      verifier.
    track_num_best_replies: Number of best tries to keep track of.
    n_bad_experience: Number of bad experiences to keep track of.
    iterate_on: Which tries to iterate on.
    initial_population_size: Number of initial population.
    initialize_from_best: Experiment path to initialize from best.
    workers: Number of workers for Initial Codegen.

  Returns:
    ConfigDict for the experiment.
  """
  req_keys = ['name', 'run', 'artifact_path']
  for k in req_keys:
    if k not in codegen_config_args:
      raise ValueError(f'Missing required key: {k}')
  # Base config args
  config = get_singlestep_config(**codegen_config_args)
  # base args for the experiment and the LM

  # Codegen args for SingleStepMultipleLMAgent: # Assigning the config for the
  # initial codegen agent, which is a
  # SingleStepMultipleLMAgent, since we start with a population of N samples.
  # This is derived from `codegen_config_args`
  model_types = [codegen_config_args['model_type']] * initial_population_size
  model_urls = [codegen_config_args['model_url']] * initial_population_size
  if 'model_config' in codegen_config_args:
    model_configs = [
        codegen_config_args['model_config']
    ] * initial_population_size
  else:
    if codegen_config_args['model_type'] in [
        'gemini_internal',
        'gemini_internal_random_seed',
    ]:
      make_default_config = (
          llm_client_builder.get_default_gemini_internal_model_config
      )
    else:
      make_default_config = (
          llm_client_builder.get_default_gemini_external_model_config
      )

    model_configs = []
    for i in range(initial_population_size):
      model_configs.append(make_default_config(i))
  template_paths = [
      codegen_config_args['template_path']
  ] * initial_population_size

  init_codegen_args = {}
  for k in [
      'version',
      'agent_class',
      'name',
      'run',
      'exp_directory',
      'artifact_path',
      'skip_existing',
  ]:
    init_codegen_args[k] = codegen_config_args[k]
  # Assign as lists
  init_codegen_args['model_types'] = model_types
  init_codegen_args['model_urls'] = model_urls
  init_codegen_args['model_configs'] = model_configs
  init_codegen_args['template_paths'] = template_paths
  init_codegen_args['save_names'] = [
      f'init_sample_{i}' for i in range(initial_population_size)
  ]

  init_codegen_args['workers'] = workers

  # Consider behavior change: make agent_class in singlestepmultilmagent
  # singlestepmultilmagent. It currently sets it as evolveragent, which is the
  # outer agent. I don't actually expect this to change behavior since nothing
  # in the SingleStepMultiLMAgent is dependent on the agent_class.

  config.initial_codegen_config = get_multiple_lm_config(**init_codegen_args)

  config.agent.initial_population_size = initial_population_size
  config.agent.correction = mlc.ConfigDict()
  config.agent.correction.subagent_class = corrector_subagent_class
  config.agent.correction.prompt = file_utils.file_open(
      corrector_prompt_path, 'r'
  ).read()
  config.agent.correction.total_tries = total_tries

  config.agent.environment = mlc.ConfigDict()
  config.agent.environment.tools = tools
  config.agent.environment.comsol_client = comsol_client
  # Whether the SubAgent should handle multiple errors at a time.
  config.agent.correction.multiple_errors = True
  config.agent.correction.save_every = save_every
  # config.agent.correction.evaluation =  Hybrid

  # Whether we plug in information retrieved by the ToolLookupAgent into context
  if tools:
    # map to langfun model
    config.agent.correction.tool_model = (
        llm_client_builder.map_model_config_to_langfun_model(config)
    )
    config.agent.correction.tool_selector_prompt = tool_selector_prompt_path
    config.agent.correction.lib_prompt_path = lib_prompt_path
    config.agent.correction.library_path = library_path
    config.agent.correction.exclude_model_ids_from_library = (
        exclude_model_ids_from_library
    )
  config.agent.correction.executability_threshold = executability_threshold

  # Best tries
  config.agent.correction.track_num_best_replies = track_num_best_replies
  config.agent.correction.n_bad_experience = n_bad_experience
  config.agent.correction.iterate_on = iterate_on
  if initialize_from_best:
    config.agent.initialize_from_best = initialize_from_best
    print('Starting with best: ', initialize_from_best)
  else:
    config.agent.initialize_from_best = ''
  return config
