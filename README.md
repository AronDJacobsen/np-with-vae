# Natural parameterised output distribution in VAEs
Course project in Deep generative modelling (Fall 2022) | DTU

# Background
Heterogeneous data consists of features of different types: real-valued numerical, nominal, ordinal data etc. More importantly, the features have different underlying probability distributions, namely a Gaussian distribution for numerical data and Bernoulli or categorical distributions for nominal and ordinal data, making it difficult to model using probabilistic generative models like Variational Autoencoders (VAEs).

These probability distributions belong to the [*exponential family*](https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions). All distributions in the *exponential family* can be written on the same form:
```math
  p(x | \boldsymbol{\eta}) = h(x) \exp{ \left( \boldsymbol{\eta}^T T(x) - A(\boldsymbol{\eta}) \right)}
``` 
parameterised by their natural parameters $`\boldsymbol{\eta}`$ and functions $`h(x)`$, $`T(x)`$ and $`A(\boldsymbol{\eta})`$. 

# Models
Models are available in the following (link)[https://drive.google.com/drive/folders/1XtPi0XuG5Kq4xLCGlalLTLzxz_0piOz-?usp=sharing]
