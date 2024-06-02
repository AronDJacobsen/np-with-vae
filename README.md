# Natural parameterised output distribution in VAEs
Course project in Deep generative modelling (Fall 2022) | DTU

# Background
Heterogeneous data contains different types of features (continuous, ordinal, nominal etc). Deep generative models, such as Variational Autoencoders (VAEs), can have a hard time learning the different underlying probability distributions for each type of feature. Common probability distributions are the Gaussian distribution for continuous features, categorical distribution for categorical features, Bernoulli distribution for binary features etc.

These probability distributions belong to the [*exponential family*](https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions). All distributions in the *exponential family* can be written on the same form:
```math
  p(x | \boldsymbol{\eta}) = h(x) \exp{ \left( \boldsymbol{\eta}^T T(x) - A(\boldsymbol{\eta}) \right)}
``` 
parameterised by their natural parameters $`\boldsymbol{\eta}`$ and functions $`h(x)`$, $`T(x)`$ and $`A(\boldsymbol{\eta})`$. We implement a VAE and modify its output distribution to be parameterised with its natural parameters.

# Distributions
|  | Gaussian | Categorical |
| :---:        |     :---:     |          :---: |
| $`\boldsymbol{\eta}`$    |$` \begin{bmatrix} \mu/\sigma^2 \\\ -1/2\sigma^2 \end{bmatrix}`$       | $` \begin{bmatrix} \log p_1 \\\ \vdots \\\ \log p_k \end{bmatrix}`$     |
| $`h(x)`$  | $`\frac{1}{2\sigma^2}`$    | 1   |
| $`T(x)`$     | $`\begin{bmatrix} x \\\ x^2\end{bmatrix}`$   |  $` \begin{bmatrix} [x = 1] \\\ \vdots \\\ [x = k] \end{bmatrix}`$     |
| $`A(\boldsymbol{\eta})`$     | $`-\frac{\eta_1^2}{4\eta_2} - \frac12 \log{(-2\eta_2)}`$    | 0    |

where $`[x=1]`$ is the Iverson bracket: $`1`$ if the expression inside is true and $`0`$ otherwise.

# Data
We benchmark the models on four [UCI datasets](https://archive.ics.uci.edu/): Avocado Sales, Bank marketing, Boston Housing, and Energy Efficiency.

# Getting started
Create conda environment 
```
conda create --name DGM python=3.9
conda activate DGM
pip install -r requirements.txt
```

Or setup environment on HPC
```
module load python3/3.9.6
module load cuda/11.7
python3 -m venv DGM
source DGM/bin/activate
pip3 install -r requirements.txt
```

# Running the model
To train and test on for instance the bank dataset, run the following or submit the shellscript ``submit.sh``.
```
python main.py --seed 3407 --device cuda --write --mode "traintest" --experiment "bank" --dataset "bank" --scale "normalize" --max_epochs 500 --max_patience 100 --prior "vampPrior" --beta 0.01
```

# Models
Models are available in the following [link](https://drive.google.com/drive/folders/1XtPi0XuG5Kq4xLCGlalLTLzxz_0piOz-?usp=sharing)
