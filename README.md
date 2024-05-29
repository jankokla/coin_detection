# Coin Detection

## Setup

### Installation

We first need to create the environment. For better reproducibility 
we're limiting the Python version to 3.9.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset

As we have flattened the original training data directory and added 
segmentation mask and coin images (extracted from training data), please 
download the full data directory from 
[here](https://drive.google.com/file/d/1Y5oaoe6CzcHp0vcF1bb3XpaTntPPkXvr/view?usp=sharing) 
and extract it to the project root.

Now you're good to go and can take a look at the `final_report.ipynb`