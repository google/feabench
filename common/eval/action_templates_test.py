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


import re

from absl.testing import absltest
from engmod.common.eval import action_templates
import numpy as np


class UtilsTest(absltest.TestCase):

  def test_create_geometry(self):
    code = 'model.component("comp1").geom().create("geom1", 3);'
    pattern = action_templates.GEOMETRY_PATTERNS['create_geometry']
    m = re.fullmatch(pattern, code)
    ans = {'dimension': '3'}
    self.assertSameStructure(m.groupdict(), ans)

  def test_create_interface(self):
    _, code, pattern = action_templates.create_interface(
        'emw', 'ElectromagneticWaves', 'geom1'
    )
    m = re.fullmatch(pattern, code)
    ans = {
        'physics_tag': 'emw',
        'name': 'ElectromagneticWaves',
        'geometry_tag': 'geom1',
    }
    self.assertSameStructure(m.groupdict(), ans)

  def test_create_feature(self):
    _, code, pattern = action_templates.create_feature(
        'emw', 'imp1', 'Impedance', 2
    )
    m = re.fullmatch(pattern, code)
    ans = {
        'physics_tag': 'emw',
        'feature_tag': 'imp1',
        'name': 'Impedance',
        'dimension': str(2),  # resolve this later
    }
    self.assertSameStructure(m.groupdict(), ans)

  def test_associate_feature_with_named_selection(self):
    _, code, pattern = action_templates.associate_feature_with_named_selection(
        'emw', 'imp1', 'sel3'
    )
    m = re.fullmatch(pattern, code)
    ans = {
        'physics_tag': 'emw',
        'feature_tag': 'imp1',
        'selection_name': 'sel3',
    }
    self.assertSameStructure(m.groupdict(), ans)

  def test_associate_feature_with_selection_number(self):
    _, code, pattern = action_templates.associate_feature_with_selection_number(
        'spf', 'inl1', '1'
    )
    m = re.fullmatch(pattern, code)
    ans = {
        'physics_tag': 'spf',
        'feature_tag': 'inl1',
        'selection_number': str(1),
    }
    self.assertSameStructure(m.groupdict(), ans)

  def test_parse_code(self):
    lines = [
        (
            'model.component("comp1").physics().create("emw",'
            ' "ElectromagneticWaves", "geom1");'
        ),
        (
            'model.component("comp1").physics("emw").create("imp1",'
            ' "Impedance", 2);'
        ),
        'model.component("comp1").physics("emw").feature("imp1").selection().named("sel1");',
        'model.component("comp1").physics("spf").feature("inl1").selection().set(1);',
        'model.component("comp1").physics("emw").create("port1", "Port", 2);',
        'model.component("comp1").physics("emw").feature("port1").selection().named("sel2");',
        (
            'model.component("comp1").physics("emw").feature("port1").set("PortType",'
            ' "Coaxial");'
        ),
        (
            'model.component("comp1").physics("mf").prop("BackgroundField").set("Bb",'
            ' new String[]{"0", "B0", "0"});'
        ),
        'model.component("comp1").physics("emw").create("port2", "Port", 2);',
        'model.component("comp1").physics("emw").feature("port2").selection().named("sel3");',
        (
            'model.component("comp1").physics("emw").feature("port2").set("PortType",'
            ' "Rectangular");'
        ),
        (
            'model.component("comp1").geometry("geom1").random("not",'
            ' "physics", "call");'
        ),
    ]

    gt_acts, template_mask = action_templates.parse_code(lines)
    ans = [
        (
            'create_interface',
            {
                'geometry_tag': 'geom1',
                'name': 'ElectromagneticWaves',
                'physics_tag': 'emw',
            },
        ),
        (
            'create_feature',
            {
                'dimension': '2',
                'feature_tag': 'imp1',
                'name': 'Impedance',
                'physics_tag': 'emw',
            },
        ),
        (
            'associate_feature_with_named_selection',
            {
                'feature_tag': 'imp1',
                'physics_tag': 'emw',
                'selection_name': 'sel1',
            },
        ),
        (
            'associate_feature_with_selection_number',
            {
                'feature_tag': 'inl1',
                'physics_tag': 'spf',
                'selection_number': '1',
            },
        ),
        (
            'create_feature',
            {
                'dimension': '2',
                'feature_tag': 'port1',
                'name': 'Port',
                'physics_tag': 'emw',
            },
        ),
        (
            'associate_feature_with_named_selection',
            {
                'feature_tag': 'port1',
                'physics_tag': 'emw',
                'selection_name': 'sel2',
            },
        ),
        (
            'modify_feature_property',
            {
                'feature_tag': 'port1',
                'param': 'PortType',
                'physics_tag': 'emw',
                'value': 'Coaxial',
            },
        ),
        (
            'modify_interface_property',
            {
                'property': 'BackgroundField',
                'physics_tag': 'mf',
                'param': 'Bb',
                'value': 'new String[]{"0", "B0", "0"}',
            },
        ),
        (
            'create_feature',
            {
                'dimension': '2',
                'feature_tag': 'port2',
                'name': 'Port',
                'physics_tag': 'emw',
            },
        ),
        (
            'associate_feature_with_named_selection',
            {
                'feature_tag': 'port2',
                'physics_tag': 'emw',
                'selection_name': 'sel3',
            },
        ),
        (
            'modify_feature_property',
            {
                'feature_tag': 'port2',
                'param': 'PortType',
                'physics_tag': 'emw',
                'value': 'Rectangular',
            },
        ),
        (None, None),
    ]
    ans_mask = np.ones(len(lines), dtype=bool)
    ans_mask[-1] = False

    self.assertTrue(np.all(template_mask == ans_mask))
    self.assertSameStructure(gt_acts, ans)


if __name__ == '__main__':
  absltest.main()
