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
from engmod.common.eval import action_eval
from engmod.common.eval import action_templates
import numpy as np


class ActionEvalTest(absltest.TestCase):
  _LM_OUTPUT1 = """[['create_interface', {'physics_tag': 'emw', 'name': 'Electromagnetic Waves', 'geometry_tag': 'geom1'}], ['create_feature', {'physics_tag': 'emw', 'feature_tag': 'imp1', 'name': 'Impedance', 'dimension': '3'}], ['modify_feature_property', {'physics_tag': 'emw', 'feature_tag': 'imp1', 'param': 'mat', 'value': 'Copper'}], ['create_feature', {'physics_tag': 'emw', 'feature_tag': 'port1', 'name': 'Port', 'dimension': '3'}], ['modify_feature_property', {'physics_tag': 'emw', 'feature_tag': 'port1', 'param': 'mode', 'value': 'TE10'}], ['modify_feature_property', {'physics_tag': 'emw', 'feature_tag': 'port1', 'param': 'direction', 'value': 'out'}], ['associate_feature_with_named_selection', {'physics_tag': 'emw', 'feature_tag': 'port1', 'selection_name': 'port1'}], ['create_feature', {'physics_tag': 'emw', 'feature_tag': 'port2', 'name': 'Port', 'dimension': '3'}], ['modify_feature_property', {'physics_tag': 'emw', 'feature_tag': 'port2', 'param': 'mode', 'value': 'TEM'}], ['modify_feature_property', {'physics_tag': 'emw', 'feature_tag': 'port2', 'param': 'direction', 'value': 'in'}], ['associate_feature_with_named_selection', {'physics_tag': 'emw', 'feature_tag': 'port2', 'selection_name': 'port2'}]]"""
  _GT_ACTIONS1 = [
      [
          'create_interface',
          {
              'geometry_tag': 'geom1',
              'name': 'ElectromagneticWaves',
              'physics_tag': 'emw',
          },
      ],
      [
          'create_feature',
          {
              'dimension': '2',
              'feature_tag': 'imp1',
              'name': 'Impedance',
              'physics_tag': 'emw',
          },
      ],
      [
          'associate_feature_with_named_selection',
          {
              'feature_tag': 'imp1',
              'physics_tag': 'emw',
              'selection_name': 'sel1',
          },
      ],
      [
          'create_feature',
          {
              'dimension': '2',
              'feature_tag': 'port1',
              'name': 'Port',
              'physics_tag': 'emw',
          },
      ],
      [
          'associate_feature_with_named_selection',
          {
              'feature_tag': 'port1',
              'physics_tag': 'emw',
              'selection_name': 'sel2',
          },
      ],
      [
          'modify_feature_property',
          {
              'feature_tag': 'port1',
              'param': 'PortType',
              'physics_tag': 'emw',
              'value': 'Coaxial',
          },
      ],
      [
          'create_feature',
          {
              'dimension': '2',
              'feature_tag': 'port2',
              'name': 'Port',
              'physics_tag': 'emw',
          },
      ],
      [
          'associate_feature_with_named_selection',
          {
              'feature_tag': 'port2',
              'physics_tag': 'emw',
              'selection_name': 'sel3',
          },
      ],
      [
          'modify_feature_property',
          {
              'feature_tag': 'port2',
              'param': 'PortType',
              'physics_tag': 'emw',
              'value': 'Rectangular',
          },
      ],
  ]

  _GT_FEATURES2 = [
      [
          'create_feature',
          {
              'dimension': '1',
              'feature_tag': 'disp1',
              'name': 'Displacement1',
              'physics_tag': 'shell',
          },
      ],
      [
          'create_feature',
          {
              'dimension': '1',
              'feature_tag': 'sym1',
              'name': 'SymmetrySolid1',
              'physics_tag': 'shell',
          },
      ],
      [
          'create_feature',
          {
              'dimension': '1',
              'feature_tag': 'disp1',
              'name': 'Displacement1',
              'physics_tag': 'solid',
          },
      ],
      [
          'create_feature',
          {
              'dimension': '2',
              'feature_tag': 'sym1',
              'name': 'SymmetrySolid',
              'physics_tag': 'solid',
          },
      ],
      [
          'create_feature',
          {
              'dimension': '2',
              'feature_tag': 'bndl1',
              'name': 'BoundaryLoad',
              'physics_tag': 'solid',
          },
      ],
      [
          'create_feature',
          {
              'dimension': '-1',
              'feature_tag': 'ge1',
              'name': 'GlobalEquations',
              'physics_tag': 'solid',
          },
      ],
      [
          'create_feature',
          {
              'dimension': '3',
              'feature_tag': 'spf1',
              'name': 'SpringFoundation3',
              'physics_tag': 'solid',
          },
      ],
  ]

  _LM_OUTPUT2 = """[['create_feature', {'physics_tag': 'shell', 'feature_tag': 'sym1', 'name': 'SymmetrySolid1', 'dimension': '1'}], ['create_feature', {'physics_tag': 'shell', 'feature_tag': 'fix1', 'name': 'Fixed', 'dimension': '3'}], ['create_feature', {'physics_tag': 'shell', 'feature_tag': 'ge1', 'name': 'GlobalEquations', 'dimension': '3'}], ['create_feature', {'physics_tag': 'shell', 'feature_tag': 'disp1', 'name': 'Displacement1', 'dimension': '3'}], ['create_feature', {'physics_tag': 'solid', 'feature_tag': 'bndl1', 'name': 'BoundaryLoad"""

  _SCORES = [
      None,
      {
          'correct_dimension_features': 20.0,
          'feature_granular_precision': 66.66666667,
      },
      {
          'correct_dimension_features': np.nan,
          'feature_granular_precision': 30.0,
      },
  ]

  def test_parsing(self):
    actions, _, _ = action_templates.convert_jsonaction_sequence_output_to_code(
        str(self._GT_ACTIONS1)
    )
    print('Expected', actions[5])
    print('Recovered', self._GT_ACTIONS1[5])
    self.assertLen(actions, len(self._GT_ACTIONS1))
    self.assertSameStructure(actions, self._GT_ACTIONS1)

  def test_metrics(self):
    evaluator = action_eval.ActionSequenceEvaluator()
    metrics = evaluator.evaluate_action_lists(
        self._LM_OUTPUT1, str(self._GT_ACTIONS1)
    )
    self.assertFalse(metrics['strict_action_types_match'])
    self.assertEqual(metrics['interface_recall'], 0)
    self.assertEqual(metrics['interface_precision'], 0)
    self.assertEqual(metrics['interface_realism'], 0)
    self.assertEqual(metrics['feature_type_recall'], 100)
    self.assertEqual(metrics['feature_type_precision'], 100)
    self.assertEqual(metrics['correct_dimension_features'], 0)
    self.assertAlmostEqual(metrics['interface_code_precision'], 0)
    self.assertAlmostEqual(metrics['interface_code_recall'], 0)

  def test_feature_metrics(self):
    evaluator = action_eval.ActionSequenceEvaluator()
    metrics = evaluator.evaluate_action_lists(
        self._LM_OUTPUT2, str(self._GT_FEATURES2)
    )
    self.assertAlmostEqual(
        metrics['feature_type_recall'], 300.0 / 6.0
    )  # BoundaryLoad, SymmetrySolid, SpringFdn missing in LM
    self.assertAlmostEqual(
        metrics['feature_type_precision'], 300.0 / 4.0
    )  # Fixed isn't in the GT
    self.assertAlmostEqual(metrics['correct_dimension_features'], 100.0 / 4.0)
    self.assertAlmostEqual(metrics['feature_granular_recall'], 250.0 / 6.0)
    self.assertAlmostEqual(metrics['feature_granular_precision'], 300.0 / 4.0)
    self.assertAlmostEqual(metrics['code_precision'], 100.0 / 5.0)
    self.assertAlmostEqual(metrics['code_recall'], 100.0 / 7.0)
    self.assertAlmostEqual(metrics['feature_code_precision'], 100.0 * 1 / 4)
    self.assertAlmostEqual(metrics['feature_code_recall'], 100.0 * 1 / 7.0)

  def test_get_mean_scores(self):
    mean_incl_bad = action_eval.get_mean_scores(
        self._SCORES,
        relevant_keys=list(self._SCORES[1].keys()),  # pylint: disable=attribute-error
        include_bad=True,
    )
    mean_only_good = action_eval.get_mean_scores(
        self._SCORES,
        relevant_keys=list(self._SCORES[1].keys()),  # pylint: disable=attribute-error
        include_bad=False,
    )
    self.assertAlmostEqual(
        mean_incl_bad['correct_dimension_features'], 20.0 / 2
    )
    self.assertAlmostEqual(mean_only_good['correct_dimension_features'], 20.0)
    self.assertAlmostEqual(
        mean_incl_bad['feature_granular_precision'], 96.66666667 / 3
    )
    self.assertAlmostEqual(
        mean_only_good['feature_granular_precision'], 96.66666667 / 2
    )


if __name__ == '__main__':
  absltest.main()
