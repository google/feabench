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
"""Run model inference on the benchmark.

python run_external_inference.py -- \
--version=0 --prompt=prompt_v0_nosol.txt --model_type=openai --run=8-28 --problems="comsol_267"
"""

import os

from absl import app
from absl import flags

from engmod.common import agent_configs
from engmod.common import constants
from engmod.common import file_utils
from engmod.common import llm_client_builder
from engmod.common import run_experiment


_VERSION = flags.DEFINE_integer(
    'version',
    None,
    'Version of the experiment. ModelSpecs=0 or Plan=1.',
    required=True,
)
_PROMPT = flags.DEFINE_string(
    'prompt',
    None,
    'Prompt to use.',
    required=True,
)
_MODEL_TYPE = flags.DEFINE_string(
    'model_type',
    None,
    'Model to test.',
    required=True,
)
_PROBLEMS = flags.DEFINE_list(
    'problems',
    list(constants.BENCHMARK_PROBLEMS),
    'Problems to run on.',
)
_RUN = flags.DEFINE_string(
    'run',
    '',
    'Results are saved under FINAL_EXPERIMENT_DIR/{name}/{model_type}_{run}/.',
)


def main(unused_argv):
  if _VERSION.value is None:
    raise ValueError('Must specify --version.')

  for prob in constants.BENCHMARK_PROBLEMS:
    assert file_utils.file_exists(
        os.path.join(constants.DATA_DIR, f'{prob}.json')
    )

  if _MODEL_TYPE.value == 'openai':
    model_url = llm_client_builder.MODEL_URLS_EXTERNAL['openai_gpt-4o']
    model_config = {'max_tokens': 8192, 'temperature': 0.0}
    # We usually need around 1k-2k in the reply, but setting this to 8192 so
    # it's identical to the Gemini experiment configs.
  elif _MODEL_TYPE.value == 'anthropic':
    model_url = llm_client_builder.MODEL_URLS_EXTERNAL['anthropic_sonnet']
    model_config = {'max_tokens': 8192, 'temperature': 0.0}
  else:
    raise ValueError(f'Unsupported model type {_MODEL_TYPE.value}')

  base_config_args = {
      'version': _VERSION.value,
      'agent_class': 'SingleStepAgent',
      'template_path': os.path.join(constants.PROMPTS_DIR, _PROMPT.value),
      'name': 'external',
      'run': _MODEL_TYPE.value + '_' + _RUN.value,
      'model_type': _MODEL_TYPE.value,
      'model_url': model_url,
      'model_config': model_config,
      'exp_directory': constants.FINAL_EXPERIMENT_DIR,
      'artifact_path': constants.SYSTEM_ARTIFACT_DIR,
      'problem_list': _PROBLEMS.value,
  }

  singlestep_config = agent_configs.get_singlestep_config(**base_config_args)
  _ = run_experiment.run_experiment(singlestep_config)


if __name__ == '__main__':
  app.run(main)
