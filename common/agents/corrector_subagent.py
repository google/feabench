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

"""Corrector Subagent and tool-calling."""

# pylint: disable=unused-import

import json
import os
from typing import Any, Callable, Optional

from engmod.common import agent_configs
from engmod.common import file_utils
from engmod.common import prompt_generation
from engmod.common.agents import tools
from engmod.common.eval import parsing_lm_utils
from engmod.common.remote_service import mph_comsol_client
import langfun as lf
import ml_collections


def render_tool_descriptions(tools_registry: dict[str, tools.Tool]) -> str:
  """Renders tool descriptions."""
  text = ''
  for name in tools_registry:
    tool = tools_registry[name]
    text += f'\n=== TOOL: {name}\n'
    text += tool.get_description()
  return text


def build_tools(
    name: str,
    comsol_client: mph_comsol_client.MphComsolClient,
    model_call_func: Callable[[str], str],
    library_path: Optional[str] = None,
) -> tools.Tool:
  """This initializes the tools."""
  if name in ['QueryPhysicsInterfaces', 'QueryPhysicsFeatures']:
    tool = tools.TOOLMAPPER[name]()
  elif name in ['QueryModelTreeProperties']:
    tool = tools.TOOLMAPPER[name](comsol_client=comsol_client)
  elif name in ['RetrieveAnnotatedSnippets']:
    tool = tools.TOOLMAPPER[name](
        model_call_func=model_call_func,
        library_path=library_path,
    )
  else:
    raise ValueError(f'Tool {name} not in TOOLMAPPER.')
  return tool


def call_tool(
    tool_call: tools.ToolCall, tools_registry: dict[str, tools.Tool]
) -> str:
  """Calls the built tool with the given arguments."""
  try:
    return tools_registry[tool_call.name].call(**tool_call.call_args)
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(
        'Tool: %s(%s) failed with error: %s'
        % (tool_call.name, tool_call.call_args, str(e))
    )
    return ''


class ToolLookupAgent(agent_configs.GenericAgent):
  """Tool Lookup Agent."""

  def __init__(
      self,
      experiment_config: ml_collections.ConfigDict,
      model_call_func: Callable[[str], str],
  ):
    super().__init__(experiment_config)
    tool_names = experiment_config.agent.environment.tools
    self.comsol_client = experiment_config.agent.environment.comsol_client
    tool_prompt_template = file_utils.file_open(
        experiment_config.agent.correction.tool_selector_prompt, 'r'
    ).read()
    self.tool_prompt_template = tool_prompt_template
    self.tool_registry = {}
    # build tool registry
    for name in tool_names:
      self.tool_registry[name] = build_tools(
          name,
          comsol_client=self.comsol_client,
          model_call_func=model_call_func,
          library_path=self.experiment_config.agent.correction.library_path,
      )
    self.tool_usage_log = []

  def update_agent(self) -> None:
    """Update tool descriptions and prompts for some tools.

    Some tool descriptions depend on the current state or the problem
    description.
    This updates QueryModelTreeProperties with the most recent model tree.

    Returns:
      None
    """
    tool_snippet = render_tool_descriptions(self.tool_registry)
    # Note: we earlier had a line that updated the tool prompt template with the
    # the problem description. Might want to add this back in?
    self.tool_lookup_prompt = prompt_generation.replace_in_prompt(
        {'{{tool_snippet}}': tool_snippet},
        self.tool_prompt_template,
    )
    return

  def run_agent(self, problem_context: dict[str, Any]) -> str:
    """Returns tool lookup response for some provided description and feedback.

    Args:
      problem_context: A dictionary containing the following keys: - 'feedback':
        The feedback from the current execution. Can either be a single line and
        error (Single-Line MainAgent) or multiple lines and their error log
        (MultipleErrors MainAgent). - 'problem_description': The problem
        description.

    Returns:
      The tool lookup response.
    """
    feedback = problem_context['feedback']

    self.update_agent()

    # Update the tool lookup prompt with the execution feedback.
    specified_prompt = prompt_generation.replace_in_prompt(
        {'{{state_info}}': feedback}, self.tool_lookup_prompt
    )
    try:
      tool_schema = lf.query(
          prompt=specified_prompt,
          schema=tools.ToolCalls,
          lm=self.experiment_config.agent.correction.tool_model,
      )
      tool_reply = ''
      # call the tools and return the combined result of the tool lookup
      for tool_call in tool_schema.list_of_tool_calls:
        tool_reply += (
            f'Tool {tool_call.name}: '
            + call_tool(tool_call, self.tool_registry)
            + '===\n'
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print('Tool Lookup Agent failed with error: %s', str(e))
      tool_reply = ''
    self.tool_usage_log.append((self.tool_lookup_prompt, feedback, tool_reply))
    return tool_reply

  def save_states(self, out_dir: str) -> None:
    """Saves the agent states to a json file."""
    with file_utils.file_open(
        os.path.join(out_dir, 'tool_lookup_agent_usage_log.json'),
        'w',
    ) as f:
      json.dump(self.tool_usage_log, f)
    for tool_name in self.tool_registry:
      tool = self.tool_registry[tool_name]
      tool.save_information(out_dir)
      # does nothing for most tools, saves the cache for
      # RetrieveAnnotatedSnippets
    return


class CorrectorSubAgentBasic(agent_configs.GenericAgent):
  """Corrector Sub Agent."""

  def __init__(
      self,
      experiment_config: ml_collections.ConfigDict,
      model_call_func: Callable[[str], str],
  ):
    super().__init__(experiment_config)
    self.model_call_func = model_call_func
    self.parser = parsing_lm_utils.CodeParser(
        parsing_lm_utils.postprocess_result
    )
    # Version: Single Error or Multiple Errors?
    self.multiple_errors = (
        self.experiment_config.agent.correction.multiple_errors
    )
    if not self.multiple_errors:
      raise NotImplementedError(
          'Single error is not implemented in this version.'
      )

    # Version: With Tools or Without Tools?
    if self.experiment_config.agent.environment.tools:
      self.tool_lookup_agent = ToolLookupAgent(
          experiment_config, model_call_func
      )
    else:
      self.tool_lookup_agent = None
    self.prompt_log = []

  def retrieve_tool_lookup(self, problem_context: dict[str, Any]) -> str:
    """Query and return the output of the tool lookup agent."""
    if self.tool_lookup_agent:
      return self.tool_lookup_agent.run_agent(problem_context)
    else:
      return ''

  def run_agent(self, problem_context: dict[str, Any]) -> str:
    """Returns a corrected code given the problem context.

    Args:
      problem_context: A dictionary containing the following keys: if
        self.multiple_errors: - 'current_code': The current code. -
        'problem_description': The problem description. - 'history': Execution
        history. - 'feedback': The feedback from the current execution. OR if
        not self.multiple_errors: - 'line': The line of code that needs to be
        corrected. - 'error': The error message associated with the line of
        code. - 'current_code': The current code.

    Returns:
      The corrected code.
    """
    problem_description = problem_context['problem_description']
    if self.multiple_errors:
      # The context includes the last N attempts, the current code, and the
      # error log.
      tool_reply = self.retrieve_tool_lookup(problem_context)
      prompt = prompt_generation.replace_in_prompt(
          {
              '{{problem}}': problem_description,
              '{{code}}': problem_context['current_code'],
              '{{history}}': problem_context['history'],
              '{{state_info}}': problem_context['feedback'],
              '{{tool_lookup}}': tool_reply,
          },
          self.experiment_config.agent.correction.prompt,
      )
    else:
      raise NotImplementedError(
          'Single error is not implemented in this version.'
      )

    self.prompt_log.append(prompt)
    reply = self.model_call_func(prompt)
    return reply

  def save_states(self, out_dir: str) -> None:
    """Saves the agent prompts used in correction thus far."""
    with file_utils.file_open(
        os.path.join(
            out_dir,
            'corrector_subagent_prompt_log.json',
        ),
        'w',
    ) as f:
      json.dump(self.prompt_log, f)
    if self.tool_lookup_agent:
      self.tool_lookup_agent.save_states(out_dir)
    return
