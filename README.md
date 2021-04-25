# Dissecting Implicit Bias in Language Models

This is the code for the Bachelor Thesis "The masculine secretary asked the visitor to sign in: Dissecting Implicit Bias in Language Models" (2019). The thesis is included in the root directory.

## Installation 

Create a virtual enviroment and install required packages.

```bash
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Scripts
Execute the scripts in the following order. Parameters have to be changed in the respective python files.

### Data generation

```bash
cd src/data_gen
python generate_winogender.py
```

### Run experiments

```bash
cd src/experiments
```

Each script can be executed separately.

#### Run Masked Language Model token predictions:

```bash
python bert_experiments.py
python bert_mlm_pronoun.py
```

#### Run Multiple Choice predictions:

```bash
python bert_multiple_choice_experiments.py
```

### Run analysis

```bash
cd src/analysis
```

Each script can be executed separately.

#### Generate Masked Language Model heatmaps:

```bash
python generate_mlm_heatmap.py
```

#### Generate Masked Language Model and Multiple Choice wordclouds:

```bash
python generate_uniques.py
python generate_wordclouds.py
```

