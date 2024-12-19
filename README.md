# Coin Detection

This repository is a group work for EPFL Masters course
***Image analysis and pattern recognition*** (EE-451) by Hanwen Zhang, Armando
Bourgknecht and Jan Kokla.

The aim of the group work was to design a robust ML system to classify coins
(EUR, CHF, other) and eventually come up with the correct count of coin types
(e.g. 1EUR, 2CHF etc) on the image.

Since it was a competition between the groups, the pre-trained NN-s had to be
trained on ImageNet-1k and the trained model no bigger than 125MB.

While we started with classical image segmentation techniques (Armando Bourgknecht),
the best results were achieved by a mixture of experts model (MoE), which consists of
several subsystems:

1. **segmentation model**: preparing better input for Hough Transform;
2. **Hough' transform**: locating the coins on the image;
3. **MoE**: first classifying currency and then specifying the coin type.

More detailed overview in the notebook `final_report.ipynb`.

## Setup

### Installation

First, create the environment. For better reproducibility
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
