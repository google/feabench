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

r"""Run external model inference on FEABench Large.

python run_external_inference_large.py -- \
--model_type=anthropic --trial=9-24 --subset=val_exec
"""

import json
import os

from absl import app
from absl import flags
from engmod.common import file_utils
from engmod.common import llm_client_builder
import pandas as pd

MODEL_TYPE = flags.DEFINE_string(
    'model_type',
    None,
    'Model to test.',
    required=True,
)
TRIAL = flags.DEFINE_string(
    'trial',
    None,
    'Results are saved under'
    ' FINAL_EXPERIMENT_DIR/{name}/{model_type}_{trial}/.',
    required=True,
)
SUBSET = flags.DEFINE_enum(
    'subset',
    default='val_exec',
    enum_values=['val_exec', 'val', 'train'],
    help='Subset of the benchmark to run on.',
)
# use below for train
MAX_RECORDS = flags.DEFINE_integer(
    'max_records',
    200,
    'Run on N records of the subset (IF you have both exec and max_records,this'
    ' filter is applied BEFORE the exec filter), so actual number will be less.'
    ' This flag was intended for the `train` subset, however.',
)

TRAIN_DIR = 'path/to/train/dir'
VAL_DIR = 'path/to/val/dir'
VAL_METADATA = 'path/to/val_metadata.csv'
TRAIN_METADATA = 'path/to/train_metadata.csv'
TRAIN_EXPDIR = 'exp_dir/for/train/'
VAL_EXPDIR = 'exp_dir/for/val/'


def run_experiment(exp_dir, task, lm_client, metadata):
  """Run experiment on some dataset.

  Args:
    exp_dir: Experiment directory to save to.
    task: Task to run.
    lm_client: LM client to use.
    metadata: Metadata about the tasks, specifically the license availability.

  Returns:
    None
  """

  modid = str(task['model_id'])
  probdir = f'comsol_{modid}/'
  savename = probdir + 'output.json'
  if not file_utils.file_exists(os.path.join(exp_dir, probdir)):
    file_utils.makedirs(os.path.join(exp_dir, probdir))
  if 'exec' in SUBSET.value:
    has_license = metadata[metadata['model_id'] == task['model_id']][
        'license_availability'
    ].item()
  else:
    has_license = True  # Want to run on all
  if file_utils.file_exists(os.path.join(exp_dir, savename)) or (
      not has_license
  ):
    print('Skipping ', task['model_id'], flush=True)
  else:
    reply = lm_client.query(task['prompt'])
    json.dump(
        {'reply': reply, 'model_id': modid},
        file_utils.file_open(os.path.join(exp_dir, savename), 'w'),
    )


def main(unused_argv):
  if SUBSET.value == 'val_exec' or SUBSET.value == 'val':
    tasks = []
    for fset in file_utils.listdir(VAL_DIR):
      tasks.extend(
          [json.loads(f) for f in file_utils.file_open(VAL_DIR + fset, 'r')]
      )
    metadata = pd.read_csv(VAL_METADATA)
    exp_dir = f'{VAL_EXPDIR}/{MODEL_TYPE.value}_{TRIAL.value}/'
    print(exp_dir)
  elif 'train' in SUBSET.value:
    tasks = []
    for fname in file_utils.listdir(TRAIN_DIR):
      if fname.startswith('comsol') and fname.endswith('.json'):
        tasks.append(
            json.load(
                file_utils.file_open(os.path.join(TRAIN_DIR, fname), 'r')
            )
        )
    metadata = pd.read_csv(TRAIN_METADATA)
    exp_dir = f'{TRAIN_EXPDIR}/{MODEL_TYPE.value}_{TRIAL.value}/'
    print(exp_dir)
  else:
    raise ValueError(f'Unsupported subset: {SUBSET.value}')

  if MODEL_TYPE.value == 'openai':
    model_config = {'max_tokens': 8192, 'temperature': 0.0}
    lm_cli = llm_client_builder.build_lm_client(
        'openai',
        llm_client_builder.MODEL_URLS_EXTERNAL['openai_gpt-4o'],
        model_config=model_config,
    )
    # print(lm_cli.query('What version are you?'))
  elif MODEL_TYPE.value == 'anthropic':
    model_config = {'max_tokens': 8192, 'temperature': 0.0}
    lm_cli = llm_client_builder.build_lm_client(
        'anthropic',
        llm_client_builder.MODEL_URLS_EXTERNAL['anthropic_sonnet'],
        model_config=model_config,
    )
  elif MODEL_TYPE.value == 'gemini_pro':
    lm_cli = llm_client_builder.build_lm_client(
        'gemini_external', 'gemini-1.5-pro-001'
    )
  else:
    raise ValueError(f'Unsupported model type {MODEL_TYPE.value}')
  print(exp_dir)
  file_utils.makedirs(exp_dir)
  for task in tasks[: MAX_RECORDS.value]:
    run_experiment(
        exp_dir=exp_dir,
        task=task,
        lm_client=lm_cli,
        metadata=metadata,
    )


if __name__ == '__main__':
  app.run(main)
