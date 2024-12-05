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

"""Constants for tools."""

# pylint: disable=trailing-whitespace
import os
from engmod.common import constants

DOC_DIR = constants.PHY_DOC_DIR
PHY_INTERFACE_PATH = os.path.join(DOC_DIR, 'physics_interfaces.json')
PHY_FEATURE_PATH = os.path.join(DOC_DIR, 'interface_feature_map.json')

# Exclude model ids in the benchmark or those in the `validation` set.
# Validation set is just a bunch of models that were earlier designated as
# validation and are meant to be excluded from any training experiments.


DESCRIPTION_ANNOTATED = """
This tool searches a library of COMSOL snippets organized by category. Use this library to retrieve example code snippets relevant to the problem context or task description for the code.
Relevant Errors: This tool can help address a wide range of errors, including:
1. Finding the complete code snippets for the study of a specific kind (eg: Stationary Study, a Parameter Sweep Study).
2. Identifying what node to create a specific feature under, especially for nested features that exist under other features.
3. Identifying what propertyGroup a specific property of a material exists under.
4. Other general errors that can be addressed by finding a snippet that matches the intent of the code.

Call_Args: {branch: str, retrieval_query: str}
`branch` must be one of ['study', 'material', 'physics', 'geometry', 'mesh', 'result', 'multiphysics', 'param', 'func', 'variable', 'misc', 'solver']. It indicates the category of snippets you want to retrieve from.
If you want to retrieve snippets that are about {selections, groups, cpl (coupling)}, select 'misc' for the branch.
`branch` can be inferred from the lines of code that you want to find relevant snippets for, as in the examples below.
`retrieval query` is a string consisting of the relevant context for your retrieval query. It can contain a description of what you would like the code you want to generate to do.
You can infer `retrieval query` from either the snippet of code that raised an error, or from code blocks that you would like to find alternate versions of.

===
EXAMPLE 0:
PROBLEM DESCRIPTION: Create a LaminarFlow interface and a coupling...Set the parameter values for the Reynolds numbers and the Viscosity of the fluid. You must then create a Rectangle with width=0.6 and height=1.0.
CODE: model.component().create("comp1", true);
model.component("comp1").mesh().create("mesh1");
model.component("comp1").physics().create("spf", "LaminarFlow", "geom1");
model.component("comp1").cpl().create("aveop1", "Average");
model.component("comp1").cpl("aveop1").set("axisym", true);
model.component("comp1").cpl("aveop1").selection().geom("geom1", 1);
model.param().label("Geometrical Parameters");
model.param().create("par2");
model.param("par2").set("Re", "100");
model.param("par2").descr("Re", "Reynolds number");
model.component("comp1").geom().create("geom1", 2);
model.component("comp1").geom("geom1").create("rect1", "Rectangle");
model.component("comp1").geom("geom1").feature("rect1").set("width", "0.6");
model.component("comp1").geom("geom1").feature("rect1").set("height", "1.0");

ERROR: model.component("comp1").geom("geom1").feature("rect1").set("width", "0.6"); -> Unknown property
Messages:
Unknown property.
- Property: width

Reasoning: The error is triggered by a geometry line. Thus `branch` will be geometry.
The code was trying to set the width of a rectangle, as is described in the problem description. Thus the retrieval query will be the CODE TASK: How to set the width of a rectangle, as well as the lines of code.
`retrieval query`

Call_Args: {branch: "geometry",
retrieval_query: Set the width of a rectangle.}
===
EXAMPLE 1`:
PROBLEM DESCRIPTION: Solve a Differential equation with a tolerance of 1e-8 from t=0 to t=100s.
CODE: ....
ERROR: The code executes, but exports an empty output file. There might be a problem with how the study code is set up, so let's try another version of the study code.

Reasoning: The `branch` for the study code is `study`.

Call_Args: {branch: "study",
retrieval_query: Set up an Optimization study for differential equations and specify the time range and tolerance.}
"""

retrieve_annotated_snippets_prompt = """You are trying to find code snippets relevant to a given concept in `QUERY`. You are provided with an annotated dictionary in LIBRARY rendered as \n `INDEX `index`: `annotation``. Identify the 3 most relevant `annotation`s and return the indices they correspond to. The index will be used to retrieve the code snippet corresponding to `annotation`. Your response should be a JSON-readable list of 3 most relevant indices.

===
Example 0:
EXAMPLE LIBRARY: INDEX 56: CONCEPTS: How to make a circle.
INDEX 38: CONCEPTS: How to make a triangle.
INDEX 299: CONCEPTS: How to make an Interval.

EXAMPLE QUERY: How to set the length of an Interval.
EXAMPLE RELEVANT INDICES: [299]

===
Now return up to 3 indices of the most relevant annotations for the query.
LIBRARY: {{lib_rendered}}
QUERY: {{query}}
RELEVANT INDICES:
"""

intent_based_toquery_prompt = """You want to look up code that matches the intent expressed in `TARGET QUERY`. Do any of the following previous queries match the target query?
Your reply should consist of the index of the previous query if you have made a similar query before, or -1, if you have not made a similar query before.
TARGET QUERY: {{target_query}}
PREVIOUS QUERIES: {{previous_queries}}
ANSWER:
"""
