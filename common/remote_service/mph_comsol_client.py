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

"""RPyC client tools for the MPh-COMSOL service."""

import json
from typing import Any, Iterator

from absl import logging
import rpyc

from retrying import retry


def _get_all_keys(x: dict[str, Any]) -> Iterator[str]:
  """Returns a generator of all the keys in a (possibly nested) dictionary."""
  for k, v in x.items():
    if isinstance(v, dict):
      yield k
      yield from _get_all_keys(v)
    else:
      yield k

_retry_wrapper =  # Specify retry function here


# Add a health-checker, block until the server is available.
class MphComsolClient:
  """Manages client side of an RPyC connection to MPh-COMSOL service."""

  def __init__(
      self,
      port: int = 18861,
      host: str = 'localhost',
      socket: str = '',
  ):
    """Connect to RPyC service."""
    if not socket:
      self._connection = rpyc.connect(host, port)
    else:
      self._connection = _retry_wrapper(
          lambda sock: rpyc.utils.factory.unix_connect(sock)  # pylint: disable=unnecessary-lambda
      )(socket)
    timeout = 900  # 15 minute timeout
    self._connection._config['sync_request_timeout'] = timeout
    logging.info('MphComsolClient timeout set to %s seconds', timeout)

  def shutdown(self):
    """Shutdown the RPyC service."""
    try:
      self._connection.root.shutdown()
    except EOFError:
      # The service already shut down.
      pass

  # Methods for managing a COMSOL mphserver
  def start_managed_mphserver(
      self,
      cores: int | None = None,
      version: str | None = None,
      port: int | None = None,
      multi: bool = True,
      timeout: int = 60,
      arguments: ... = None,
  ) -> str:
    """Start a COMSOL mphserver on the server."""
    status_message = self._connection.root.start_managed_mphserver(
        cores=cores,
        version=version,
        port=port,
        multi=multi,
        timeout=timeout,
        arguments=arguments,
    )
    return status_message

  def mphserver_port(self) -> int:
    """Port number the mphserver is listening on for client connections."""
    return self._connection.root.mphserver_port()

  def mphserver_running(self) -> bool:
    """Returns whether the mpherver process is still running."""
    return self._connection.root.mphserver_running()

  def mphserver_stop(self, timeout: int = 20) -> str:
    """Shuts down the mphserver."""
    status_message = self._connection.root.mphserver_stop(timeout=timeout)
    return status_message

  def health_check(self) -> bool:
    """Checks if the compute engine and the license manager have started."""
    return self._connection.root.health_check()

  # Methods managing a COMSOL API Client
  def connect_api_client_to_managed_mphserver(self) -> str:
    """Create an API client connected to the managed mphserver."""
    status_message = (
        self._connection.root.connect_api_client_to_managed_mphserver()
    )
    return status_message

  def connect_api_client_to_mphserver(
      self, port: int, host: str = 'localhost'
  ) -> str:
    """Create an API client connected to an mphserver."""
    status_message = self._connection.root.connect_api_client_to_mphserver(
        port=port, host=host
    )
    return status_message

  def disconnect_api_client(self) -> None:
    """Disconnects the client from the server."""
    self._connection.root.disconnect_api_client()

  def licensed_modules(self) -> list[str]:
    """Returns the names of available licensed COMSOL modules."""
    return self._connection.root.licensed_modules()

  def model_names(self) -> list[str]:
    """Returns the names of all loaded models."""
    return self._connection.root.model_names()

  def model_files(self) -> list[str]:
    """Returns the file-system paths of all loaded models."""
    return self._connection.root.model_files()

  def load_model(self, path: str) -> None:
    """Load model and set it to the current model."""
    self._connection.root.load_model(path)

  def create_model(self, name: str) -> None:
    """Create a new model and set it to the current model."""
    self._connection.root.create_model(name=name)

  def set_model(self, name: str) -> None:
    """Set the current model."""
    self._connection.root.set_model(name=name)

  def remove_model(self, name: str) -> None:
    """Remove model from memory."""
    self._connection.root.remove_model(name=name)

  def clear_models(self) -> None:
    """Remove all loaded models from memory."""
    self._connection.root.clear_models()

  # Methods acting on the COMSOL model
  def model_name(self) -> str:
    """Returns the model name."""
    return self._connection.root.model_name()

  def model_file(self) -> str:
    """Returns the absolute path to the file the model was loaded from."""
    return self._connection.root.model_file()

  def model_save(
      self,
      path: str | None = None,
      format: str | None = None,  # pylint: disable=redefined-builtin
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
    self._connection.root.model_save(path=path, format=format)

  def model_api_call(self, expression: str | list[str]) -> str | list[str]:
    """Apply one or more COMSOL Java API expression to the model."""
    return self._connection.root.model_api_call(expression=expression)

  def model_groups(self) -> list[str]:
    """Returns the names of built-in groups found in all models."""
    return self._connection.root.model_groups()

  def model_tree(self, node_path: str = '') -> str:
    """Returns the model tree."""
    return self._connection.root.model_tree(node_path=node_path)

  def model_tree_attribute(
      self,
      node_path: str = '',
      attribute: str = 'name',
  ) -> dict[str, Any]:
    """Returns the model tree attribute as a nested dict."""
    return json.loads(
        self._connection.root.model_tree_attribute(
            node_path=node_path, attribute=attribute
        )
    )

  def model_tree_mapping(
      self, node_path: str = '', key_attr: str = 'tag', value_attr: str = 'path'
  ) -> dict[str, str]:
    """Returns mapping from one model tree node attribute to another."""
    tags = self.model_tree_attribute(node_path, key_attr)
    paths = self.model_tree_attribute(node_path, value_attr)
    return dict(zip(_get_all_keys(tags), _get_all_keys(paths)))

  def model_remove_all_nodes(self) -> None:
    """Remove all nodes from model (root and groups remain)."""
    return self._connection.root.model_remove_all_nodes()

  def node_name(self, path: str) -> str:
    """Returns the node name."""
    return self._connection.root.node_name(path=path)

  def node_tag(self, path: str) -> str:
    """Returns the node tag."""
    return self._connection.root.node_tag(path=path)

  def node_type(self, path: str) -> str:
    """Returns the node feature type."""
    return self._connection.root.node_type(path=path)

  def node_parent(self, path: str) -> str:
    """Returns the parent node path."""
    return self._connection.root.node_parent(path=path)

  def node_children(self, path: str) -> list[str]:
    """Returns all child node paths."""
    return self._connection.root.node_children(path=path)

  def node_exists(self, path: str) -> bool:
    """Returns True if the node exists in the model tree."""
    return self._connection.root.node_exists(path=path)

  def node_properties(self, path: str) -> dict[str, str]:
    """Returns names and values of all node properties as a dictionary."""
    return self._connection.root.node_properties(path=path)

  def node_selection(self, path: str) -> list[int] | dict[str, str]:
    """Returns the entity or entities the node has selected."""
    return self._connection.root.node_selection(path=path)

  def get_file_contents(self, filepath: str) -> str:
    """Returns the contents of a file, or an error message."""
    return self._connection.root.get_file_contents(filepath=filepath)

  def makedirs(self, path: str) -> str:
    """Makes a directory on the filesystem visible to the COMSOL client."""
    return self._connection.root.makedirs(path=path)

  def file_exists(self, path: str) -> bool:
    """Returns True if the file exists on the filesystem."""
    return self._connection.root.file_exists(path=path)

  def bash(self, command: str) -> str:
    """Gives direct command-line access to the kernel, for debugging."""
    return self._connection.root.bash(command=command)
