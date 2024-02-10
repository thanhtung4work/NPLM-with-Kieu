# That NPLM Model

This repository implements a Neural Probabilistic Language Model (NPLM) based on the research paper "A Neural Probabilistic Language Model" by Yoshua Bengio, Réjean Houle, Nicolas Le Roux, and Michaël Ouimet (2003).

## Installation
```python
pip install -r requirements.txt
```

## Model Description

This NPLM uses neural networks to capture the statistical relationships between words in a given language. It associates each word with a unique "feature vector" that encodes its semantic meaning and relationships to other words. By analyzing the sequence of these vectors, the model can predict the next word in a sentence with high probability.

## Citation

Bengio, Y., Houle, R., Le Roux, N., & Ouimet, M. (2003). A neural probabilistic language model. Journal of machine learning research, 3(1), 1137-1155.