# FEABench: Evaluating Language Models on Real-World Physics Reasoning Ability

[**FEABench: Evaluating Language Models on Real-World Physics Reasoning Ability**](https://arxiv.org/abs/2411.xxxxx)

Nayantara Mudur, Hao Cui, Subhashini Venugopalan, Paul Raccuglia, Michael Brenner, Peter Norgaard

## Datasets
The datasets are organized as follows:
### Benchmark / Evaluation Datasets
1. [FEABench Gold](hflink)
2. [FEABench Large](hflink): This is a CSV file containing the application ids, URLs and titles of the 200 problems we evaluated on. The code to generate the inputs and outputs is in `engmod/generate_feabench_large`.

### Library of Annotated Snippets
This is a dataset with the following structure
```bash
theme
  ├── model_id
  ├── annotation
  └── snippet
```



## 📁 Repository Structure

The directories in this repo are organized as follows:

```bash
engmod
    ├── common
        ├── agents  # Code pertaining to the Corrector and ToolLookup `subagents` and Tools.
        ├── eval  # Code to evaluate results
        └── remote_service  # Code to set up the MPHClient
    ├── generate_feabench_large  # Code to segment tutorial pdfs and JAVA files
```

## Inference Workflow

**Single-Turn Evaluation**
Specify directory locations in `engmod/common/constants.py`

On FEABench Gold
```bash
python run_external_inference.py -- \
--version=0 --prompt=prompt_v0_nosol.txt --model_type=openai --run=8-28 --problems="comsol_267"
```

Specify directory locations in `engmod/common/constants.py`

On FEABench Large
```bash
python run_external_inference_large.py -- \
--model_type=anthropic --trial=9-24 --subset=val
```
