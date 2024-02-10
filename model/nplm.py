import torch
from torch import nn

class NPLM(nn.Module):
    """
    Neural Probabilistic Language Model (NPLM) class for word prediction.

    Parameters:
    - vocab_size (int): Size of the vocabulary.
    - emb_dim (int): Dimension of word embeddings.
    - hidden_size (int): Size of the hidden layer.
    - ngram (int): Size of the context window for n-gram modeling.

    Attributes:
    - emb_dim (int): Dimension of word embeddings.
    - ngram (int): Size of the context window for n-gram modeling.
    - C (nn.Embedding): Embedding layer for input words.
    - H (nn.Parameter): Parameter matrix for the hidden layer transformation.
    - d (nn.Parameter): Parameter vector for the hidden layer bias.
    - U (nn.Parameter): Parameter matrix for the output layer transformation.
    - W (nn.Parameter): Parameter matrix for the context layer transformation.
    - b (nn.Parameter): Parameter vector for the output layer bias.

    Methods:
    - forward(x): Forward pass of the NPLM model.

    Usage:
    nplm_model = NPLM(vocab_size, emb_dim, hidden_size, ngram)
    output = nplm_model(input_tensor)
    """
    def __init__(self, vocab_size, emb_dim, hidden_size, ngram):
        super().__init__()
        self.emb_dim = emb_dim
        self.ngram = ngram

        self.C = nn.Embedding(vocab_size, emb_dim)
        self.H = nn.Parameter(
            torch.randn((ngram - 1) * emb_dim, hidden_size)
        )
        self.d = nn.Parameter(torch.randn(hidden_size))
        self.U = nn.Parameter(
            torch.randn(hidden_size, vocab_size)
        )
        self.W = nn.Parameter(
            torch.randn((ngram - 1) * emb_dim, vocab_size)
        )
        self.b = nn.Parameter(torch.randn(vocab_size))


    def forward(self, x):
        """
        Forward pass of the NPLM model.

        Parameters:
        - x (torch.Tensor): Input tensor representing word indices.

        Returns:
        torch.Tensor: Output tensor representing the predicted word probabilities.
        """
        x = self.C(x)
        x = x.view(-1, self.emb_dim * (self.ngram - 1))

        Hx = torch.mm(x, self.H)
        UHx = torch.mm(
            torch.tanh(self.d + Hx), self.U
        )
        Wx = torch.mm(x, self.W)
        return self.b + Wx + UHx

    