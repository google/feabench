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

"""RPyC service for MPh-COMSOL.

Intended usage uses the MPh library to start a COMSOL mphserver on localhost and
connects a COMSOL API client.  COMSOL models can then be created, loaded, saved,
etc. and the entire COMSOL API is accessible.
"""

import contextlib
import io
import json
import os
import signal
from typing import Any
import mph
import numpy as np
import rpyc


class MphComsolService(rpyc.Service):
  """An RPyC service for an MPh-COMSOL interface.

  Attributes:
    server: Manages a Comsol mphserver process.
    client: Manages the Comsol API client instance.
    model: Represents a Comsol model.
  """

  def __init__(self):
    super(MphComsolService, self).__init__()
    self.server = None
    self.client = None
    self.model = None

  def on_connect(self, conn):
    # code that runs when a connection is created
    # (to init the service, if needed)
    pass

  def on_disconnect(self, conn):
    # code that runs after the connection has already closed
    # (to finalize the service, if needed)
    pass

  def exposed_shutdown(self):
    """Shutdown the RPyC service."""
    pid = os.getpid()
    os.kill(pid, signal.SIGTERM)

  ### Methods for mph.Server ###
  def exposed_start_managed_mphserver(
      self,
      cores=None,
      version=None,
      port=None,
      multi=True,
      timeout=60,
      arguments=None,
  ):
    """Start a managed COMSOL mphserver."""
    if self.server is not None:
      raise RuntimeError(
          f'An mphserver is already running on port {self.server.port}.'
      )
    try:
      self.server = mph.Server(
          cores=cores,
          version=version,
          port=port,
          multi=multi,
          timeout=timeout,
          arguments=arguments,
      )
      return (
          f'Started a COMSOL mphserver: {cores=}, {version=}, {port=}, on'
          f' subprocess {self.server.process}'
      )
    except RuntimeError as e:
      return e

  def exposed_mphserver_port(self):
    """Port number the mphserver is listening on for client connections."""
    if self.server is None:
      return None
    return self.server.port

  def exposed_mphserver_running(self):
    """Returns whether the mpherver process is still running."""
    if self.server is None:
      return False
    return self.server.running()

  def exposed_mphserver_stop(self, timeout=20):
    """Shuts down the mphserver."""
    if self.server is None:
      return 'No COMSOL mphserver running.'
    port = self.server.port
    self.server.stop(timeout=timeout)
    self.server = None
    return f'Stopped COMSOL mphserver on {port=}'

  ### Methods for mph.Client ###
  def exposed_connect_api_client_to_managed_mphserver(self):
    """Create an API client connected to the managed mphserver."""
    if self.server is None:
      raise RuntimeError('No mphserver is running.')
    self.exposed_connect_api_client_to_mphserver(
        port=self.server.port, host='localhost'
    )

  def exposed_connect_api_client_to_mphserver(self, port, host='localhost'):
    """Create an API client connected to an mphserver."""
    if self.client is None:
      self.client = mph.Client(port=port, host=host)
    else:
      try:
        self.client.connect(port=port, host=host)
      except NotImplementedError as e:
        print(e)
        self.client.disconnect()
        self.client.connect(port=port, host=host)
    return f'COMSOL API client connected to port {port}'

  def exposed_disconnect_api_client(self):
    if self.client is None:
      raise RuntimeError('No API client exists.')
    self.client.disconnect()

  def exposed_licensed_modules(self):
    """Returns the names of available licensed COMSOL modules."""
    if self.client is None:
      return RuntimeError('No API client exists.')
    return self.client.modules()

  def exposed_model_names(self):
    """Returns the names of all loaded models."""
    if self.client is None:
      return RuntimeError('No API client exists.')
    return self.client.names()

  def exposed_model_files(self):
    """Returns the file-system paths of all loaded models."""
    if self.client is None:
      return RuntimeError('No API client exists.')
    return self.client.files()

  def exposed_load_model(self, path):
    """Load model and set it to the current model."""
    self.model = self.client.load(path)

  def exposed_create_model(self, name):
    """Create a new model and set it to the current model."""
    if name in self.exposed_model_names():
      raise ValueError(
          f'Model {name=} already exists in {self.exposed_model_names()}'
      )
    self.model = self.client.create(name)

  def _get_model_index(self, name):
    try:
      return self.exposed_model_names().index(name)
    except ValueError:
      return f'model "{name}" not found in {self.exposed_model_names()}'

  def exposed_set_model(self, name):
    """Set the current model."""
    idx = self._get_model_index(name)
    self.model = self.client.models()[idx]

  def exposed_remove_model(self, name):
    """Remove model from memory."""
    idx = self._get_model_index(name)
    model = self.client.models()[idx]
    self.client.remove(model)
    return f'model "{name}" removed'

  def exposed_clear_models(self):
    """Remove all loaded models from memory."""
    self.client.clear()

  ### Methods for mph.Model ###
  def exposed_model_name(self):
    """Returns the model name."""
    if self.model is None:
      return None
    return self.model.name()

  def exposed_model_file(self):
    """Returns the absolute path to the file the model was loaded from."""
    if self.model is None:
      return None
    return self.model.file()

  def exposed_model_save(
      self, path: str | None = None, format: str | None = None
  ) -> None:
    """Saves the model at the given file-system path.

    Args:
      path: File-system path for saving the model file. If path is not given,
        the original file name is used, i.e. the one from which the model was
        loaded to begin with. If the path contains no folder information, the
        current folder (working directory) is used. If the path points to a
        folder, the model name is used to name the file inside that folder.
      format: A format can be specified as either “Comsol”, “Java”, “Matlab”, or
        “VBA”. If no format is given, it will be deduced from the file's ending,
        being either .mph, .java, .m, or .vba, respectively. No file ending
        implies “Comsol” format.

    Imposes the correct file ending for the format. Overwrites existing files.
    """
    self.model.save(path=path, format=format)

  def _model_api_call(self, expression):
    """Apply one COMSOL Java API expression to the model."""
    if self.model is None:
      return RuntimeError('No Model set.')
    # Define the local variable `model` to enable eval() of strings of the form
    #   model.comp("comp1").geom("geom1").create(...)
    model = self.model.java  # pylint: disable=unused-variable
    # Protect against general python expressions
    if not expression.startswith('model.'):
      raise ValueError('expression must be of the form model.x')
    # Catch COMSOL API client exceptions and return message
    try:
      output = eval(expression)  # pylint: disable=eval-used
    except Exception as e:  # pylint: disable=broad-exception-caught
      return str(e)
    if output is None:
      return ''
    return str(output)

  def exposed_model_api_call(self, expression):
    """Apply one or more COMSOL Java API expression to the model."""
    if isinstance(expression, str):
      return self._model_api_call(expression)
    else:
      return [self._model_api_call(x) for x in expression]

  def exposed_model_groups(self) -> list[str]:
    """Returns the names of built-in groups found in all models."""
    return list((self.model / '').groups.keys())

  def exposed_model_tree(self, node_path: str = '') -> str:
    """Returns the model tree."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
      mph.tree(self.model / node_path)
    return buffer.getvalue()

  def exposed_model_tree_attribute(
      self, node_path: str = '', attribute: str = 'name'
  ) -> dict[str, Any]:
    """Returns the model tree node attribute as a nested dict."""

    def get_attribute(node, attribute):
      if attribute == 'name':
        return node.name()
      elif attribute == 'tag':
        return node.tag()
      elif attribute == 'path':
        return '/'.join(node.path)
      else:
        raise ValueError(
            'Invalid attribute, should be one of {"name", "tag", "path"}'
        )

    def get_children_attribute(path):
      this_node = self.model / path
      children = this_node.children()
      if not children:
        return ''
      return {
          get_attribute(node, attribute): get_children_attribute(
              '/'.join(node.path)
          )
          for node in children
      }

    # Use json dumps since rpyc netref dict does not expose .items()
    # https://github.com/tomerfiliba-org/rpyc/issues/272
    return json.dumps(
        {
            get_attribute(
                self.model / node_path, attribute
            ): get_children_attribute(node_path)
        }
    )

  def _model_remove_nodes(self, path: str = '') -> None:
    """Remove this nodes and all children (excluding root and groups)."""
    this_node = self.model / path
    for node in this_node.children():
      path = '/'.join(node.path)
      self._model_remove_nodes(path)
    if path == 'parameters/Parameters 1':
      return  # special case node that cannot be deleted
    if not this_node.is_root() and not this_node.is_group():
      this_node.remove()

  def exposed_model_remove_all_nodes(self) -> None:
    """Remove all nodes from model (root and groups remain)."""
    self._model_remove_nodes()
    self._model_remove_nodes()

  def exposed_node_name(self, path: str) -> str:
    """Returns the node name."""
    node = self.model / path
    if not node.exists():
      return f'Node {path} does not exist'
    return node.name()

  def exposed_node_tag(self, path: str) -> str:
    """Returns the node tag."""
    return (self.model / path).tag()

  def exposed_node_type(self, path: str) -> str:
    """Returns the node feature type."""
    return (self.model / path).type()

  def exposed_node_parent(self, path: str) -> str:
    """Returns the parent node path."""
    node = self.model / path
    if not node.exists():
      return f'Node {path} does not exist'
    return '/'.join((self.model / path).parent().path)

  def exposed_node_children(self, path: str) -> list[str]:
    """Returns all child node paths."""
    node = self.model / path
    if not node.exists():
      return f'Node {path} does not exist'
    return ['/'.join(c.path) for c in node.children()]

  def exposed_node_exists(self, path: str) -> bool:
    """Returns True if the node exists in the model tree."""
    return (self.model / path).exists()

  def exposed_node_properties(self, path: str) -> dict[str, str]:
    """Returns names and values of all node properties as a dictionary."""
    return (self.model / path).properties()

  def exposed_node_selection(self, path: str) -> list[int] | dict[str, str]:
    """Returns the entity or entities the node has selected."""
    node = self.model / path
    if node.is_root() or node.is_group():
      return []  # model root and groups have no selection
    selection = node.selection()
    if isinstance(selection, np.ndarray):
      return list(selection)  # numeric selection
    return {
        'name': selection.name(),
        'tag': selection.tag(),
        'path': selection.path,
    }  # named selection

  # Exposes too much control over the service.
  def exposed_debugging_evaluation(self, expression):
    model = self.model.java  # pylint: disable=unused-variable
    return eval(expression)  # pylint: disable=eval-used

  def exposed_get_file_contents(self, filepath: str) -> str:
    try:
      with open(filepath, 'r') as f:
        return f.read()
    except FileNotFoundError as e:
      return f'FileNotFoundError: {e}'

  def exposed_makedirs(self, path: str) -> str:
    try:
      os.makedirs(path, exist_ok=True)
    except OSError as e:
      return f'OSError: {e}'

  def exposed_file_exists(self, path: str) -> str:
    return os.path.exists(path)

if __name__ == '__main__':
  rpyc_port = 18861
  t = rpyc.utils.server.ThreadedServer(MphComsolService, port=rpyc_port)
  print(f'Started MPh-COMSOL RPyC service on port {rpyc_port}')
  t.start()
