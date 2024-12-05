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

"""Utils and code related to building LLM clients."""

from typing import Any, Optional
import langfun as lf
import numpy as np
from retrying import retry
from engmod import llm_client
from engmod.common import constants


# @retry() # Specify retry decorator here
def retry_wrapper(func, input_dict):
  return func(**input_dict)


MODEL_URLS_EXTERNAL = {
    'openai_gpt-4o': 'gpt-4o-2024-08-06',
    'anthropic_sonnet': 'claude-3-5-sonnet-20240620',
}


def get_default_gemini_external_model_config(seed: int) -> dict[str, Any]:
  """Returns the default model config for the given model type."""
  return {
      'max_output_tokens': 8192,
      'temperature': 1.0,
      'seed': seed,
  }


class LLMClient:
  """Base class for LLM clients."""

  def query(self, prompt: str) -> str:
    raise NotImplementedError()


class VertexClientRandomSeed(LLMClient):
  """Vertex client for LLM calls."""

  def __init__(
      self,
      service_account: str,
      project_id: str,
      location: str,
      model_name: str,
      model_config: dict[str, Any],  # Note: model_config is no longer optional
  ) -> Any:
    if not model_config:
      model_config = get_default_gemini_external_model_config(0)
    self.lm_client = llm_client.VertexClient(
        service_account=service_account,
        project_id=project_id,
        location=location,
        model_name=model_name,
        model_config=model_config,
    )
    self.rng = np.random.default_rng(seed=42)
    self.model_config = model_config

  def query(self, prompt: str, seed: Optional[int] = None) -> str:
    """Queries with either an explicit seed or by randomizing it."""
    if seed:
      sub_key = seed
    else:
      sub_key = self.rng.choice(1000)
    print(sub_key)
    self.model_config['seed'] = sub_key
    return self.lm_client.query(prompt, self.model_config)


def build_lm_client(
    model_type: str,
    model_url: str,
    model_config: Any | None = None,
) -> LLMClient | llm_client.LLMClient:
  """Make client."""
  if model_type == 'gemini_external':
    return llm_client.VertexClient(
        service_account=constants.VERTEX_SERVICE_ACCT,
        project_id=constants.VERTEX_PROJECT_ID,
        location=constants.VERTEX_LOCATION,
        model_name=model_url,
        model_config=model_config,
    )
  elif model_type == 'gemini_external_random_seed':
    return VertexClientRandomSeed(
        service_account=constants.VERTEX_SERVICE_ACCT,
        project_id=constants.VERTEX_PROJECT_ID,
        location=constants.VERTEX_LOCATION,
        model_name=model_url,
        model_config=model_config,
    )
  elif model_type == 'anthropic':
    api_key = constants.get_api_key(model_type)
    return llm_client.AnthropicClient(
        model_name=model_url, api_key=api_key, **model_config
    )
  elif model_type == 'openai':
    # This calls the langfun wrapper around the OpenAI client.
    # Querying OpenAI directly raised a trawler error.
    api_key = constants.get_api_key('openai')
    return llm_client.OpenAILFClient(
        model_name=model_url, api_key=api_key, **model_config
    )
  else:
    raise ValueError('Unsupported model type: %s' % model_type)


def map_model_config_to_langfun_model(experiment_config):
  """Maps model config to langfun model."""
  model_type, model_url, model_config = (
      experiment_config.agent.language_model.type,
      experiment_config.agent.language_model.model_url,
      experiment_config.agent.language_model.model_config,
  )
  if model_type in ['gemini_external', 'gemini_external_random_seed']:
    if model_config is None:
      model_config = get_default_gemini_external_model_config(seed=0)
    if model_url in ['gemini-1.5-pro-001', 'gemini-1.5-flash-001']:
      return lf.llms.VertexAIGeminiPro1_5(
          project=constants.VERTEX_PROJECT_ID,
          location=constants.VERTEX_LOCATION,
          temperature=model_config['temperature'],
          max_tokens=model_config['max_output_tokens'],
      )
  else:
    raise NotImplementedError(
        f'Implement the mapping for model type: {model_type}'
    )
