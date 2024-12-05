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
from absl.testing import parameterized

import parse_java_api


class ParseComsolApiTest(parameterized.TestCase):

  def test_separate_run_method_body(self):
    model_java_api = """/*
 * my_model.java
 */

import com.comsol.model.*;
import com.comsol.model.util.*;

/** Model exported on ... */
public class my_model {

  public static Model run() {
    Model model = ModelUtil.create("Model");

    model.modelPath("/some/path/my_model");

    model.component().create("comp1", true);

    model.label("my_model.mph");

    return model;
  }

  public static void main(String[] args) {
    run();
  }

}
"""
    contents = parse_java_api.separate_run_method_body(model_java_api)
    self.assertEqual(
        contents['before'],
        """/*
 * my_model.java
 */

import com.comsol.model.*;
import com.comsol.model.util.*;

/** Model exported on ... */
public class my_model {

  public static Model run() {
""",
    )

    self.assertEqual(
        contents['body'],
        """    Model model = ModelUtil.create("Model");

    model.modelPath("/some/path/my_model");

    model.component().create("comp1", true);

    model.label("my_model.mph");

    return model;
""",
    )

    self.assertEqual(
        contents['after'],
        """  }

  public static void main(String[] args) {
    run();
  }

}
""",
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='compactify_non_mph_lines',
          input_code="""    Model model = ModelUtil.create("Model");

    model.modelPath("/path/to/my_model");

    model.component().create("comp1", true);

    model.component("comp1").geom().create("geom1", 3);

    return model;
""",
          expected_code="""model.component().create("comp1", true);\nmodel.component("comp1").geom().create("geom1", 3);\n""",
          expected_removed_lines={
              'model_creation': (
                  '    Model model = ModelUtil.create("Model");\n'
              ),
              'model_path': '    model.modelPath("/path/to/my_model");\n',
              'return_model': '    return model;\n',
          },
      ),
      dict(
          testcase_name='compactify_line_continuation',
          input_code="""    model.component("comp1").geom("geom1").feature("cyl4").
         set("pos", new int[]{0, 0, 12});""",
          expected_code="""model.component("comp1").geom("geom1").feature("cyl4").set("pos", new int[]{0, 0, 12});\n""",
          expected_removed_lines={},
      ),
      dict(
          testcase_name='compactify_multiexpression_lines',
          input_code="""  model.component("comp1").geom("geom1").run(); model.second_expression();\n""",
          expected_code="""model.component("comp1").geom("geom1").run();\nmodel.second_expression();\n""",
          expected_removed_lines={},
      ),
  )
  def test_compactify_java(
      self, input_code, expected_code, expected_removed_lines
  ):
    body, removed_lines = parse_java_api.compactify_java(input_code)
    self.assertEqual(expected_code, body)
    self.assertEqual(expected_removed_lines, removed_lines)

  @parameterized.named_parameters(
      dict(
          testcase_name='no args',
          input_line='model.component("comp1").geom("geom1").run();',
          expected='',
      ),
      dict(
          testcase_name='single arg',
          input_line='model.study().create("std1");',
          expected='"std1"',
      ),
      dict(
          testcase_name='integer list',
          input_line=(
              'model.component("comp1").geom("geom1").feature("cyl2")'
              '.set("pos", new int[]{0, 0, 8});'
          ),
          expected='"pos", new int[]{0, 0, 8}',
      ),
      dict(
          testcase_name='string list',
          input_line=(
              'model.sol("sol1").feature("s1").feature("p1")'
              '.set("plistarr", new String[]{"1 10 100 1000 10000 100000"});'
          ),
          expected='"plistarr", new String[]{"1 10 100 1000 10000 100000"}',
      ),
      dict(
          testcase_name='string arg with whitespace',
          input_line=(
              '    model.component("comp1").geom("geom1").feature("cyl2")'
              '.setIndex("layername", "Layer 1", 0);'
          ),
          expected='"layername", "Layer 1", 0',
      ),
      dict(
          testcase_name='geom() method at end of expression',  # Corner case.
          input_line=(
              '    model.component("comp1").view("view1").hideEntities("hide1")'
              '.geom("geom1", 2);'
          ),
          expected='"geom1", 2',
      ),
      dict(
          testcase_name='empty string',
          input_line=(
              'model.result().numerical("gev2").setIndex("descr", "", 0);'
          ),
          expected='"descr", "", 0',
      ),
  )
  def test_get_api_argstring(self, input_line, expected):
    actual = parse_java_api._get_api_argstring(input_line)
    self.assertEqual(expected, actual)

  def test_get_api_argstring_no_semicolon(self):
    line = 'model.component("comp1").geom("geom1").run()'
    with self.assertRaisesRegex(ValueError, 'Expected java line to end with'):
      parse_java_api._get_api_argstring(line)

  @parameterized.named_parameters(
      dict(
          testcase_name='no args',
          input_argstring='',
          expected='',
      ),
      dict(
          testcase_name='single string arg',
          input_argstring='"std1"',
          expected='"std1"',
      ),
      dict(
          testcase_name='single int arg',
          input_argstring='1',
          expected='1',
      ),
      dict(
          # Single int arg should be never be converted to string.
          testcase_name='disambiguate single int arg',
          input_argstring='1',
          expected='1',
          convert_overloaded_types=True,
      ),
      dict(
          testcase_name='single float arg',
          input_argstring='1.2',
          expected='1.2',
      ),
      dict(
          testcase_name='disambiguate single float arg',
          input_argstring='1.2',
          expected='"1.2"',
          convert_overloaded_types=True,
      ),
      dict(
          testcase_name='numeric args',
          input_argstring='1, 2., 3.1',
          expected='1, 2.0, 3.1',
      ),
      dict(
          testcase_name='disambiguate numeric args',
          input_argstring='1, 2., 3.1',
          expected='"1", "2.0", "3.1"',
          convert_overloaded_types=True,
      ),
      dict(
          testcase_name='bool args',
          input_argstring='true, false',
          expected='True, False',
      ),
      dict(
          testcase_name='disambiguate bool args',
          input_argstring='true, false',
          expected='"on", "off"',
          convert_overloaded_types=True,
      ),
      dict(
          testcase_name='integer list',
          input_argstring='"pos", new int[]{0, 0, 8}',
          expected='"pos", [0, 0, 8]',
      ),
      dict(
          testcase_name='disambiguate integer list',
          input_argstring='"pos", new int[]{0, 0, 8}',
          expected='"pos", ["0", "0", "8"]',
          convert_overloaded_types=True,
      ),
      dict(
          testcase_name='string list',
          input_argstring='"geom1", 3, 2, new String[]{"exterior"}',
          expected='"geom1", 3, 2, ["exterior"]',
      ),
      dict(
          testcase_name='math formula',
          input_argstring='"expr", "imag(ec.Y11)/ec.omega", 0',
          expected='"expr", "imag(ec.Y11)/ec.omega", 0',
      ),
      dict(
          testcase_name='string with whitespace',
          input_argstring='"descr", "Volumetric loss density, electric", 0',
          expected='"descr", "Volumetric loss density, electric", 0',
      ),
      dict(
          testcase_name='empty string',
          input_argstring='"descr", "", 0',
          expected='"descr", "", 0',
      ),
      dict(
          testcase_name='multidim arrays',
          input_argstring=(
              'new String[]{"a", "b"}, new double[][]{{1.1, 2., 3}}'
          ),
          expected='["a", "b"], [[1.1, 2.0, 3]]',
      ),
      dict(
          testcase_name='disambiguate multidim arrays',
          input_argstring=(
              'new String[]{"a", "b"}, new double[][]{{1.1, 2., 3}}'
          ),
          expected='["a", "b"], [["1.1", "2.0", "3"]]',
          convert_overloaded_types=True,
      ),
  )
  def test_pythonize_argstring(
      self,
      input_argstring,
      expected,
      convert_overloaded_types=False,
  ):
    actual = parse_java_api._pythonize_argstring(
        input_argstring,
        convert_overloaded_types,
    )
    self.assertEqual(expected, actual)

  @parameterized.named_parameters(
      dict(
          testcase_name='no args',
          input_line='model.component("comp1").geom("geom1").run();',
          expected='model.component("comp1").geom("geom1").run()',
      ),
      dict(
          # Cast java bool to python bool.
          testcase_name='create() with bool',
          input_line='model.component().create("comp1", true);',
          expected='model.component().create("comp1", True)',
      ),
      dict(
          # create() expects (str, int).
          testcase_name='create() with int',
          input_line='model.component("comp1").geom().create("geom1", 3);',
          expected='model.component("comp1").geom().create("geom1", 3)',
      ),
      dict(
          # set() with one arg expects int.
          testcase_name='set() selection with int',
          input_line=(
              'model.component("comp1").material("mat2").selection().set(2);'
          ),
          expected=(
              'model.component("comp1").material("mat2").selection().set(2)'
          ),
      ),
      dict(
          # selections().set() with int first arg expects all int args.
          testcase_name='set() selection with multiple int',
          input_line=(
              'model.component("comp1").material("mat2").selection().set(2, 3);'
          ),
          expected=(
              'model.component("comp1").material("mat2").selection().set(2, 3)'
          ),
      ),
      dict(
          # set() with string first arg expects str second arg.
          testcase_name='disambiguate set() with int cast to str',
          input_line=(
              'model.component("comp1").geom("geom1").feature("cyl1")'
              '.set("r", 20);'
          ),
          expected=(
              'model.component("comp1").geom("geom1").feature("cyl1")'
              '.set("r", "20")'
          ),
      ),
      dict(
          # set() throws "ambiguous overload" error for (str, bool).
          # Special case requires true/false translated to "on"/"off".
          testcase_name='disambiguate set() with bool cast to "on"/"off"',
          input_line=(
              'model.component("comp1").view("view1")'
              '.set("renderwireframe", true);'
          ),
          expected=(
              'model.component("comp1").view("view1")'
              '.set("renderwireframe", "on")'
          ),
      ),
      dict(
          # set() throws "ambiguous overload" error for (str, bool).
          # Special case requires true/false translated to "0"/"1".
          testcase_name=(
              'disambiguate set() with bool cast to "0"/"1"--exception:0/1'
          ),
          input_line=(
              'model.component("comp1").physics("gop").feature("matd1").feature("film1").set("RepeatLayerInMultilayerFilms",'
              ' false);'
          ),
          expected=(
              'model.component("comp1").physics("gop").feature("matd1").feature("film1").set("RepeatLayerInMultilayerFilms",'
              ' "0")'
          ),
      ),
      dict(
          # Convert java array to python lists.
          # For set() with str first arg, second arg should be (list of) str.
          testcase_name='args include int array',
          input_line=(
              'model.component("comp1").geom("geom1").feature("cyl4")'
              '.set("pos", new int[]{0, 0, 12});'
          ),
          expected=(
              'model.component("comp1").geom("geom1").feature("cyl4")'
              '.set("pos", ["0", "0", "12"])'
          ),
      ),
      dict(
          # Convert java multidim array to nested python lists.
          # For set() with str first arg, second arg should be (list of) str.
          testcase_name='args include multidim string array',
          input_line=(
              'model.component("comp1").material("mat1").propertyGroup("def").func("eta").set("pieces",'
              ' new String[][]{{"200.0", "1600.0",'
              ' "-8.38278E-7+8.35717342E-8*T^1-7.69429583E-11*T^2+4.6437266E-14*T^3-1.06585607E-17*T^4"}});'
          ),
          expected=(
              'model.component("comp1").material("mat1").propertyGroup("def").func("eta").set("pieces",'
              ' [["200.0", "1600.0",'
              ' "-8.38278E-7+8.35717342E-8*T^1-7.69429583E-11*T^2+4.6437266E-14*T^3-1.06585607E-17*T^4"]])'
          ),
      ),
      dict(
          # Always cast second arg of setIndex() to str, remaining args int.
          testcase_name='setIndex() uses (str, str, ...)',
          input_line='model.func("an1").setIndex("plotargs", 20, 0, 1);',
          expected='model.func("an1").setIndex("plotargs", 20.0, 0, 1)',
      ),
      dict(
          # Always cast second arg of setIndex() to str, remaining args int.
          testcase_name='setIndex() casts to (str, float, int, int): Case 1',
          input_line='model.func("an1").setIndex("table", 2, 0, 1);',
          expected='model.func("an1").setIndex("table", 2.0, 0, 1)',
      ),
      dict(
          # Always cast second arg of setIndex() to str, remaining args int.
          testcase_name='setIndex() casts to (str, float, int, int): Case 2',
          input_line='model.func("an1").setIndex("table", 2.1, 0, 1);',
          expected='model.func("an1").setIndex("table", 2.1, 0, 1)',
      ),
      dict(
          # Always cast second arg of setIndex() to str, remaining args int.
          testcase_name='setIndex() casts to (str, float, int): Case 1',
          input_line=(
              'model.component("comp1").physics("c").feature("cfeq1").setIndex("da",'
              ' -1, 0);'
          ),
          expected=(
              'model.component("comp1").physics("c").feature("cfeq1").setIndex("da",'
              ' -1.0, 0)'
          ),
      ),
      dict(
          # Always cast second arg of setIndex() to str, remaining args int.
          testcase_name='setIndex() casts to (str, int, int): Case 1',
          input_line=(
              'model.component("comp1").physics("beam").feature("pdr1").setIndex("FreeRotationAround",'
              ' true, 1);'
          ),
          expected=(
              'model.component("comp1").physics("beam").feature("pdr1").setIndex("FreeRotationAround",'
              ' 1.0, 1)'
          ),
      ),
  )
  def test_pythonize_java_api(self, input_line, expected):
    actual = parse_java_api.pythonize_java_api(input_line)
    self.assertEqual(expected, actual)


if __name__ == '__main__':
  absltest.main()
