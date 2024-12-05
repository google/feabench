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

"""Base class for LLM clients."""

from typing import Any, Optional, Union

from google.cloud.aiplatform import aiplatform
from google.cloud.aiplatform import vertexai
from google.cloud.aiplatform.vertexai.preview import generative_models
import langfun as lf
import numpy as np
import retry_lib


def _get_credentials(scope: str, service_account=None):
  """Gets credentials for the service account."""
  pass


class LLMClient:
  """Base class for LLM clients."""

  def query(self, prompt: str) -> str:
    raise NotImplementedError()


class OpenAILFClient(LLMClient):
  """Client for querying Openai endpoints via langfun."""

  def __init__(
      self,
      model_name: str,
      api_key: str,
      temperature: float = 0.0,
      top_k: float = 0.1,
      top_p: float = 0.2,
      stop: list[str] | None = None,
      max_tokens: int | None = None,
  ):
    """Initializes the Openai client."""
    self.model_name = model_name
    self.api_key = api_key
    self.llm_engine = lf.core.llms.OpenAI(model=model_name, api_key=api_key)
    self.temperature = temperature
    self.top_k = top_k
    self.top_p = top_p
    self.stop = stop or ['\n']
    self.max_tokens = max_tokens

  def query(self, prompt: str) -> str:
    """Queries Openai model for a given prompt."""
    return str(
        lf.query(
            prompt,
            lm=self.llm_engine,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop,
            max_tokens=self.max_tokens,
        )
    )


class AnthropicClient(LLMClient):
  """Client for querying Anthropic inference endpoints."""

  def __init__(
      self,
      model_name: str,
      api_key: str,
      temperature: float = 0.0,
      top_k: float = 0.1,
      top_p: float = 0.2,
      stop: list[str] | None = None,
      max_tokens: int | None = None,
  ):
    """Initializes the Anthropic client."""
    self.model_name = model_name
    self.api_key = api_key
    self.llm_engine = lf.core.llms.Anthropic(model=model_name, api_key=api_key)
    self.temperature = temperature
    self.top_k = top_k
    self.top_p = top_p
    self.stop = stop or ['\n']
    self.max_tokens = max_tokens

  def query(self, prompt: str) -> str:
    """Queries Anthropic model for a given prompt."""
    return str(
        lf.query(
            prompt,
            lm=self.llm_engine,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            stop=self.stop,
            max_tokens=self.max_tokens,
        )
    )


class VertexClient(LLMClient):
  """Client for querying Vertex AI endpoints."""

  def __init__(
      self,
      service_account: str,
      project_id: str,
      location: str,
      model_name: str,
      model_config: dict[str, Any] | None = None,
  ):
    """Initializes the client.

    Args:
      service_account: a Google Cloud service account, with permission to query
        Vertex. The user must be configured with Service Account Token Creator
        to use this account.
      project_id: the Google Cloud project id to bill.
      location: the data center location (e.g., us-west1) to use
      model_name: the vertex model id. E.g., gemini-1.5-pro-preview-0409.
      model_config: a dict of generationconfig parameters.
    """
    self.service_account = service_account
    self.project_id = project_id
    self.location = location
    self.model_name = model_name

    aiplatform.init(
        credentials=_get_credentials(
            'https://www.googleapis.com/auth/cloud-platform',
            service_account=(service_account),
        )
    )

    vertexai.init(project=project_id, location=location)

    self.model = generative_models.GenerativeModel(model_name=model_name)
    self.model_config = model_config
    if model_config:
      for k in model_config:
        if k == 'temperature' and not isinstance(model_config[k], float):
          raise TypeError('Temperature must be a float.')
        if k == 'max_output_tokens' and not isinstance(model_config[k], int):
          raise TypeError('max_output_tokens must be an int')
        if k == 'seed' and not isinstance(
            model_config[k], Union[np.integer, int]
        ):
          raise TypeError('Seed must be an int.', model_config[k])

  def _query_vertex(
      self, prompt: str, generation_config: Optional[dict[str, Any]] = None
  ) -> str:
    """Queries Vertex AI. Returns the model response."""
    response = self.model.generate_content(
        prompt, generation_config=generation_config
    )
    if not isinstance(response, generative_models.GenerationResponse):
      raise ValueError(
          'Unexpected response type. Expecting GenerationResponse, got'
          f' {response}.'
      )
    if not response.candidates:
      raise ValueError('No candidates returned.')

    parts = []
    for part in response.candidates[0].content.parts:
      parts.append(part.text)
    return ' '.join(parts)

  def query(
      self,
      prompt: str,
      generation_config: Optional[dict[str, Any]] = None,
      max_attempts: Optional[int] = 8,
  ) -> str:
    """Queries Vertex AI. Returns the model response. Uses backoff to retry."""
    if not generation_config:
      generation_config = self.model_config
    return retry_lib.call_function_with_retry(
        lambda: self._query_vertex(prompt, generation_config), max_attempts
    )
