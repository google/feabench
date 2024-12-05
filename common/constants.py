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

"""Benchmark constants."""
from engmod.common import file_utils

DATA_DIR = # DATADIR 

LIB_DIR = # LIB_DIR for Annotated Library
PROMPTS_DIR = 
SYSTEM_ARTIFACT_DIR = # Local filepath that mphclient can see
WORKING_EXPERIMENT_DIR = # EXP_DIR for outputs
FINAL_EXPERIMENT_DIR = # EXP_DIR for outputs

LIB_PATH = f'{LIB_DIR}/annotated_snippets_v1.json'

BENCHMARK_PROBLEMS = (
    'comsol_82',
    'comsol_75611',
    'comsol_266',
    'comsol_723',
    'comsol_265',
    'comsol_453',
    'comsol_12351',
    'comsol_12677',
    'comsol_12681_gravity',
    'comsol_12681_force',
    'comsol_19275',
    'comsol_1863',
    'comsol_267',
    'comsol_10275',
    'comsol_16635',
)

# VertexAI const
VERTEX_SERVICE_ACCT = 
VERTEX_PROJECT_ID = 
VERTEX_LOCATION = 

ANTHROPIC_PATH = 
OPENAI_PATH = 


def get_api_key(model_type: str) -> str:
  if model_type == 'anthropic':
    return file_utils.file_open(ANTHROPIC_PATH, 'r').read().strip()
  elif model_type == 'openai':
    return file_utils.file_open(OPENAI_PATH, 'r').read().strip()
  else:
    raise ValueError('Unsupported model type: %s' % model_type)
