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

from absl.testing import absltest
from engmod.common.eval import utils


class UtilsTest(absltest.TestCase):

  def test_render_code(self):
    replies = [
        """Type: Component
          Tag: comp1
          Identifier: comp1
          Line 1:model.component("comp1").geom().create("geom1", 2) -> Type: Geometry
          Tag: geom1
          Dimension: 2
          Current child:
          Child nodes: fin""",
        """Ambiguous overloads found for com.comsol.clientapi.impl.PropFeatureClient.setIndex(str,int,int,int) between:
      public com.comsol.model.PropFeature com.comsol.clientapi.impl.PropFeatureClient.setIndex(java.lang.String,boolean,int,int)
      public com.comsol.model.PropFeature com.comsol.clientapi.impl.PropFeatureClient.setIndex(java.lang.String,int,int,int)
    """,
        """Exception:
      com.comsol.util.exceptions.FlException: Unknown physics interface feature
    Messages:
      Unknown physics interface feature.
      - Tag: out1""",
        """Uncaught Error Message""",
        """invalid syntax""",
        """Exception: Ambiguous""",
        """No Model set""",
    ]
    flags = utils.render_code_by_correctness(replies, replies)
    exp_flags = [
        'Correct',
        'Translation error',
        'Syntax error',
        'Correct',
        'Syntax error',
        'Syntax error',
        'Translation error',
    ]
    self.assertEqual(flags, exp_flags)


if __name__ == '__main__':
  absltest.main()
