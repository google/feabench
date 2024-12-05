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

"""Tools for the Corrector SubAgent."""

# pylint: disable=broad-exception-caught
# pylint: disable=g-doc-args

import json
import os
from typing import Any, Literal

from engmod import llm_client
from engmod.common import constants
from engmod.common import file_utils
from engmod.common.agents import tools_constants
from google.cloud import discoveryengine_v1alpha as discoveryengine
import pyglove as pg


class Tool:
  """Base class for tools."""

  description: str

  def __init__(self, **kwargs):
    raise NotImplementedError('Implement __init__() in the subclass.')

  def call(self, **kwargs) -> str:
    raise NotImplementedError('Implement call() in the subclass.')

  def get_description(self) -> str:
    """Unless overwritten, returns the static description."""
    return self.description

  def save_information(self, out_dir: str) -> None:
    """Saves information about what the tool did. Default: do nothing."""


# Specific tools
class QueryPhysicsInterfaces(Tool):
  """Tool that returns the list of valid physics interfaces in COMSOL."""

  def __init__(self, **kwargs):
    self.interfaces = str(
        json.load(
            file_utils.file_open(tools_constants.PHY_INTERFACE_PATH, 'r')
        )
    )
    self.description = """
This tool returns the list of valid physics interfaces available in COMSOL.
Relevant Errors: "Invalid physics interface".
Call_Args: {}"""

  def call(self, **kwargs) -> str:
    return 'The list of valid physics interfaces is: ' + self.interfaces


class QueryPhysicsFeatures(Tool):
  """Tool that returns the list of valid physics features in COMSOL."""

  def __init__(self, **kwargs):
    self.features = json.load(
        file_utils.file_open(tools_constants.PHY_FEATURE_PATH, 'r')
    )
    intface2tags = {k: self.features[k]['tags'] for k in self.features}
    self.description = (
        """
This tool returns the list of valid physics features available under each physics interface in COMSOL.
Only a feature that is available under a physics interface can be created under that interface. This function can be called with an optional interface argument to return the features available under that interface.
Relevant Errors: "Unknown feature".
Call_Args: {interface: str}
You can infer the name of the interface from the tag (the 2-3 letter phrase) of the parent interface node. The interface argument must be an interface in this mapping:
"""
        + str(intface2tags)
        + """\n ===EXAMPLE:
FEEDBACK:
LINE:  model.component("comp1").physics("patd").create("pp1", "PipePressure", 2);
ERROR: Unknown feature ID. PipePressure.

REASONING: The interface is "patd". This corresponds to the TransientPipeAcoustics interface in the mapping.
Call_Args: {interface: "TransientPipeAcoustics"}
===
"""
    )

  def call(self, **kwargs) -> str:
    interface = kwargs.pop('interface', None)
    if interface and interface in self.features:
      rel_features = str(self.features[interface]['features'])
      return (
          'The features available under the interface '
          + interface
          + ' are: '
          + rel_features
          + '.'
      )
    else:
      return 'The features available under all interfaces are: ' + str(
          self.features
      )


class QueryModelTreeProperties(Tool):
  """Tool that returns the properties available under a node in the model tree."""

  def __init__(self, **kwargs):
    comsol_client = kwargs.pop('comsol_client')
    self.comsol_client = comsol_client
    self.mapping = json.load(
        file_utils.file_open(tools_constants.PHY_FEATURE_PATH, 'r')
    )
    self.description = ''
    self.memory = {'good_paths': [], 'bad_paths': []}
    # Technically good and bad paths depend on the model tree but just ignoring
    # that for now, since the model tree doesn't evolve * that * much over time.

  def call(self, **kwargs) -> str:
    path = kwargs.pop('path')
    try:
      reply = f'The properties available under the node {path} are:'
      reply += '\n'
      node_props = self.comsol_client.node_properties(path).__str__()
      self.memory['good_paths'].append(path)
      return reply + node_props
    except (AttributeError, LookupError, Exception):
      self.memory['bad_paths'].append(path)
      return f'Invalid Model Tree Path: {path}'

  def get_description(self) -> str:
    model_tree = (
        self.comsol_client.model_tree()
        if self.comsol_client.model_name()
        else 'Empty Model'
    )
    desc = (
        """
This tool returns the properties available under a node, such as a feature or interface in the COMSOL model tree, along with the default values assigned to the properties.
Relevant Errors: Unknown parameter, Unknown property.
Call_Args: {path: str}
The path MUST be a substring that is present in CURRENT MODEL TREE, where each branch under a parent node will now be separated by a '/'.
You can infer the path to the node that triggered the error from the model tree and the mapping of interfaces and features to their tags, as in the example below.

INTERFACE_FEATURE_MAP:
"""
        + str(self.mapping)
        + """\nCURRENT MODEL TREE: """
        + str(model_tree)
        + """\n
===EXAMPLE:
FEEDBACK:
LINE:  model.component("comp1").physics("emw").feature("port1").set("Z0", "50");
ERROR: Unknown parameter Z0.
CURRENT MODEL TREE:
Example Tree:
├─ parameters
│  └─ Parameters 1
├─ functions
│  ├─ Interpolation
├─ physics
│  └─ Electromagnetic Waves, Frequency Domain
│     ├─ Lumped Port 1
│...

REASONING:
1) All paths must start with the top level branches of the MODEL TREE, i.e. '/parameters', '/functions', '/physics' for the tree above.
Valid paths from the tree above are:
`/parameters/Parameters 1`
`/functions/Interpolation`
`/physics/Electromagnetic Waves, Frequency Domain`
`/physics/Electromagnetic Waves, Frequency Domain/Lumped Port 1`

2) The error message indicates that the parameter is not found under the feature tagged as `port1`, which must be `Lumped Port 1` in the tree
The path is to query is thus:
`/physics/Electromagnetic Waves, Frequency Domain/Lumped Port 1`.

Call_Args: {path: "/physics/Electromagnetic Waves, Frequency Domain/Lumped Port 1"}
===
"""
    )
    if self.memory['good_paths'] or self.memory['bad_paths']:
      mem_render = """From recent calls, here are some paths:"""
      mem_render += (
          """Valid Paths: """
          + str(self.memory['good_paths'])
          + """\nInvalid Paths: """
          + str(self.memory['bad_paths'])
      )
    else:
      mem_render = ''
    return desc + mem_render


class Library:
  """Library of code snippets for each concept."""

  def __init__(
      self,
      concept_library_path: str,
  ):
    print('Using library path: ', concept_library_path)
    if file_utils.file_exists(concept_library_path):
      with file_utils.file_open(concept_library_path, 'r') as f:
        self.concept_library = json.load(f)
    else:
      raise ValueError(
          'Concept library path does not exist: ' + concept_library_path
      )
      # Note: we had an implementation where it loads and makes the library of
      # code snippets on the fly for the non-annotated code snippets.
      # Not adding this here to reduce clutter, if we anyway plan to use the
      # the annotated code snippet library for now.


def parse_into_list(inp: str, as_list: bool = True) -> list[int] | str:
  """Parses the output of the model into a list of integers."""
  start = inp.rindex('[')
  end = inp.rindex(']')
  inp = inp[start + 1 : end]
  if as_list:
    stripped = inp.split(',')
    return [int(elem.replace("'", '').replace('"', '')) for elem in stripped]
  else:
    return inp


class RetrieveAnnotatedSnippets(Tool):
  """Tool that queries a library of branch wise annotated code snippets."""

  def __init__(self, **kwargs):
    library_path = kwargs.pop('library_path')

    self.num_examples = 3
    self.cache = {}
    self.description = tools_constants.DESCRIPTION_ANNOTATED
    self.retriever = discoveryengine.RankServiceClient(
        credentials=llm_client._get_credentials(
            'https://www.googleapis.com/auth/cloud-platform',
            service_account=(constants.VERTEX_SERVICE_ACCT),
        )
    )
    self.ranking_config = self.retriever.ranking_config_path(
        project=constants.VERTEX_PROJECT_ID,
        location='global',
        ranking_config='default_ranking_config',
    )
    self.branched_library = Library(concept_library_path=library_path)
    # organize as records
    self.library = {}
    max_records = 200
    for branch in self.branched_library.concept_library:
      sublib = self.branched_library.concept_library[branch]
      records = []
      for model_id in self.branched_library.concept_library[branch]:
        records.append(
            discoveryengine.RankingRecord(
                id=model_id, content=sublib[model_id]['summary']
            )
        )
      if len(records) > max_records:
        print('Truncating library for branch', branch)
        records = records[:max_records]
      self.library[branch] = records

  def call(self, **kwargs) -> str:
    """Calls the tool to retrieve relevant code snippets.

    This may reuse older cached snippets and first uses a query to check whether
    retrieval_query matches a previously made query.

    Args:
      branch: The branch of the library to query.
      retrieval_query: The query to use to retrieve relevant code snippets.

    Returns:
      A string containing the relevant code snippets.
    """
    branch = kwargs.pop('branch')
    retrieval_query = kwargs.pop('retrieval_query')
    try:
      if branch not in self.library:
        raise ValueError(f'Branch {branch} not found in library.')
      request = discoveryengine.RankRequest(
          ranking_config=self.ranking_config,
          model='semantic-ranker-512-003',
          top_n=self.num_examples,
          query=retrieval_query,
          records=self.library[branch],
      )
      reply = self.retriever.rank(request=request)
      print('Branch: ', branch)
      print('Retriever: ', retrieval_query)
      print('Top Snippet: ', reply.records[0].id, reply.records[0].content)
      prefix = f"""Here are some code snippets relevant to {branch} code for {retrieval_query}.\n"""
      for retrieved_record in reply.records:
        modid = retrieved_record.id
        summ = retrieved_record.content
        code = '\n'.join(
            self.branched_library.concept_library[branch][modid]['code']
        )
        prefix += f'{summ}:\nCode: {code}\n===\n'
      snippets = prefix
      if branch not in self.cache:
        self.cache[branch] = {retrieval_query: snippets}
      else:
        self.cache[branch][retrieval_query] = snippets
      return snippets
    except Exception as e:
      print('Failed to retrieve annotated snippets', str(e))
      return ''

  def save_information(self, out_dir: str) -> None:
    """Saves the cache to a json file."""
    with file_utils.file_open(
        os.path.join(out_dir, 'tool_retrieve-annotated_cache.json'),
        'w',
    ) as f:
      json.dump(self.cache, f)


TOOLMAPPER = {
    'QueryPhysicsInterfaces': QueryPhysicsInterfaces,
    'QueryPhysicsFeatures': QueryPhysicsFeatures,
    'QueryModelTreeProperties': QueryModelTreeProperties,
    'RetrieveAnnotatedSnippets': RetrieveAnnotatedSnippets,
}


# Langfun Schema Objects
class ToolCall(pg.Object):
  name: Literal[
      'QueryPhysicsInterfaces',
      'QueryPhysicsFeatures',
      'QueryModelTreeProperties',
      'RetrieveAnnotatedSnippets',
  ]
  call_args: dict[str, Any] | None


class ToolCalls(pg.Object):
  list_of_tool_calls: list[ToolCall]
