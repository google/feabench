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


import dataclasses

from absl.testing import absltest
from engmod.common.eval import evaluate_bench
import numpy as np


EMPTY_TABLE = """
% Model:              Untitled.mph
% Version:            COMSOL 6.1.0.357
% Date:               Jul 17 2024, 02:34
% Table:              Table 1 - Point Evaluation 1

"""

SUCCESSFUL_TABLE_1COL = """
% Model:              Untitled.mph
% Version:            COMSOL 6.1.0.357
% Date:               Jul 20 2024, 07:50
% Table:              Table 1 - Point Evaluation 1
% Stress tensor, yy-component (MPa), Point: (0.001, 0)
-49.349603652932565
"""

SUCCESSFUL_TABLE_2COL = """
% Model:              Untitled.mph
% Version:            COMSOL 6.1.0.357
% Date:               Jul 17 2024, 02:34
% Table:              Table 1 - Point Evaluation 1
% Time (s)               Lift Coeff (1/m^2)
5                        -0.8070670878701368
"""


class EvaluateNafemsTest(absltest.TestCase):

  def test_empty_table(self):
    table, error, _ = evaluate_nafems.evaluate_table_output(EMPTY_TABLE, 0.3)
    self.assertLen(table.table_metadata, 4)
    self.assertEmpty(table.table_data)
    self.assertEqual(table.column_headers, "")
    self.assertEqual(table.last_row, "")
    self.assertTrue(np.isnan(table.last_value))
    self.assertTrue(np.isnan(error))

  def test_1column_table(self):
    table, error, _ = evaluate_nafems.evaluate_table_output(
        SUCCESSFUL_TABLE_1COL, -49.34
    )
    self.assertLen(table.table_metadata, 5)
    self.assertLen(table.table_data, 1)
    self.assertEqual(
        table.column_headers,
        "% Stress tensor, yy-component (MPa), Point: (0.001, 0)",
    )
    self.assertEqual(table.last_row, "-49.349603652932565")
    self.assertAlmostEqual(table.last_value, -49.349603652932565)
    self.assertLess(error, 0.001)

  def test_2column_table(self):
    table, error, _ = evaluate_nafems.evaluate_table_output(
        SUCCESSFUL_TABLE_2COL, -0.807
    )
    self.assertLen(table.table_metadata, 5)
    self.assertLen(table.table_data, 1)
    self.assertEqual(
        table.column_headers,
        "% Time (s)               Lift Coeff (1/m^2)",
    )
    self.assertEqual(
        table.last_row, "5                        -0.8070670878701368"
    )
    self.assertAlmostEqual(table.last_value, -0.8070670878701368)
    self.assertLess(error, 0.001)
    table_dict_version = dataclasses.asdict(table)
    self.assertLen(table_dict_version["table_metadata"], 5)
    self.assertLen(table_dict_version["table_data"], 1)
    self.assertEqual(
        table_dict_version["column_headers"],
        "% Time (s)               Lift Coeff (1/m^2)",
    )
    self.assertEqual(
        table_dict_version["last_row"],
        "5                        -0.8070670878701368",
    )
    self.assertAlmostEqual(
        table_dict_version["last_value"], -0.8070670878701368
    )


if __name__ == "__main__":
  absltest.main()
