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

"""Tools for parsing tutorial documentation pdfs and tex."""

import dataclasses
import pypdfium2 as pdfium
import regex as re


@dataclasses.dataclass(frozen=True)
class TutorialDoc:
  """Tutorial PDF.

  Attributes:
    model_id: Application Gallery ID
    version:
    source_path: Path to the pdf
    parse_method: Method used to parse the pdf
    title: Title of the tutorial
    introduction: Introduction section
    model_definition: Model definition section
    results_and_discussion: Results and discussion section
    references: References section
    modeling_instructions: Modeling instructions section
  """

  model_id: int = -1
  version: str = ""
  source_path: str = ""
  parse_method: str = ""
  title: str = ""
  introduction: str = ""
  model_definition: str = ""
  results_and_discussion: str = ""
  references: str = ""
  modeling_instructions: str = ""


@dataclasses.dataclass(frozen=True)
class Section:
  """Tutorial sections.

  The suffix `alt` denotes alternative section names found in some docs.
  """

  title = "Title\r\n"
  intro = "Introduction\r\n"
  model_def = "Model Definition\r\n"
  results = "Results and Discussion\r\n"
  results_alt1 = "Results\r\n"
  references = "References\r\n"
  references_alt1 = "Reference\r\n"
  modeling_instr = "Modeling Instructions\r\n"
  modeling_instr_alt1 = "Model Instructions\r\n"
  modeling_instr_alt2 = "Modeling Instructions — COMSOL Desktop\r\n"


def _pdfium_parse_page_text(pdf_data: bytes) -> list[str]:
  """Parses PDF data with pdfium and returns the text from each page."""
  pdf_doc = pdfium.PdfDocument(pdf_data)
  page_text = [page.get_textpage().get_text_range() for page in pdf_doc]
  return page_text


def _remove_footer(page_text: list[str]) -> list[str]:
  """Remove the footer text from each page."""
  page_text_no_footer = [page_text[0]]  # no footer on page 1
  for idx, page_text in enumerate(page_text[1:]):
    page_number = idx + 2  # first page in this iteration is 2
    pattern = r'{page_number} \| [A-Z0-9 -]*'.format(page_number=page_number)
    footer = re.search(pattern, page_text).group()
    if not footer:
      raise ValueError(f"No footers found on page {page_number}: {page_text}")
    page_text = page_text.replace(footer, "")
    page_text_no_footer.append(page_text)
  return page_text_no_footer


def _combine_page_text(page_text: list[str]) -> str:
  """Combine the text from each page into a single string."""
  doc_text = "".join(page_text)
  return doc_text


def _required_sections_exist(doc_text: str) -> bool:
  """Returns true if all required tutorial sections exist in doc_text."""
  has_intro = Section.intro in doc_text
  has_model_def = Section.model_def in doc_text
  has_modeling_instr = (
      Section.modeling_instr in doc_text
      or Section.modeling_instr_alt1 in doc_text
      or Section.modeling_instr_alt2 in doc_text
  )
  # Note, references not required
  exists = {
      Section.intro: has_intro,
      Section.model_def: has_model_def,
      Section.modeling_instr: has_modeling_instr,
  }
  if not all(exists.values()):
    raise ValueError(f"Missing required sections in doc_text: {exists}")
  return True


def _split_into_sections(doc_text: str) -> dict[str, str]:
  """Returns a dictionary of tutorial sections from doc_text."""
  split = lambda text, name: text.split(name, maxsplit=1)
  right_split = lambda text, name: text.rsplit(name, 1)

  # Split off the title page and the introduction from the start of the doc.
  # These always exist.
  title, intro_etc = split(doc_text, Section.intro)
  intro, model_def_etc = split(intro_etc, Section.model_def)

  # Split off modeling instructions from the end of the doc.
  # This always exists, but there are some format alternatives.
  # The remaining text contains the Model Definition, Results, and References.
  if Section.modeling_instr in model_def_etc:
    model_def_result_ref, modeling_instr = right_split(
        model_def_etc, Section.modeling_instr
    )
  elif Section.modeling_instr_alt1 in model_def_etc:
    model_def_result_ref, modeling_instr = right_split(
        model_def_etc, Section.modeling_instr_alt1
    )
  elif Section.modeling_instr_alt2 in model_def_etc:
    model_def_result_ref, modeling_instr = right_split(
        model_def_etc, Section.modeling_instr_alt2
    )
  else:
    raise ValueError(
        f"Valid Modeling Instructions section text not found in {model_def_etc}"
    )

  # Split the References off the end.
  # Handle alternative formats, or case that References does not exist.
  if Section.references in model_def_result_ref:
    model_def_results, references = right_split(
        model_def_result_ref, Section.references
    )
  elif Section.references_alt1 in model_def_result_ref:
    model_def_results, references = right_split(
        model_def_result_ref, Section.references_alt1
    )
  else:
    model_def_results = model_def_result_ref
    references = ""

  # Split off the Results, if it exists.  Handle alternative formats.
  if Section.results in model_def_results:
    model_def, results = split(model_def_results, Section.results)
  elif Section.results_alt1 in model_def_etc:
    model_def, results = split(model_def_results, Section.results_alt1)
  else:
    model_def = model_def_results
    results = ""

  return {
      Section.title: title,
      Section.intro: intro,
      Section.model_def: model_def,
      Section.results: results,
      Section.references: references,
      Section.modeling_instr: modeling_instr,
  }


def _parse_tutorial_doc(
    page_text: list[str],
) -> TutorialDoc:
  """Take the text from each page and parse into the text for each section."""
  page_text_no_footer = _remove_footer(page_text)
  doc_text = _combine_page_text(page_text_no_footer)
  assert _required_sections_exist(doc_text)
  section_text = _split_into_sections(doc_text)

  tutorial_doc = TutorialDoc(
      title=section_text[Section.title],
      introduction=section_text[Section.intro],
      model_definition=section_text[Section.model_def],
      results_and_discussion=section_text[Section.results],
      references=section_text[Section.references],
      modeling_instructions=section_text[Section.modeling_instr],
  )
  return tutorial_doc


def parse_with_pdfium(
    pdf_data: bytes,
) -> TutorialDoc:
  """Given a tutorial pdf, parse into the text for each section.

  The tutorial docs follow a template, and are parsed with this expectations.

  Args:
    pdf_data: The byte string from reading a pdf file.

  Returns:
    TutorialDoc: Contains the text from each section of the tutorial pdf.
  """
  page_text = _pdfium_parse_page_text(pdf_data)
  return _parse_tutorial_doc(page_text)
