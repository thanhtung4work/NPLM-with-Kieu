import numpy as np
import torch
from torch import nn, optim

from dataset import KieuCorpus
from model import NPLM


def train(model, criterion, optimizer, dataloader, epoch=10):
    for epoch in range(epoch):
        for context, target in dataloader:
            output = model(context)
            loss = criterion(output, target)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, loss: {loss.item():.3f}")


if __name__ == "__main__":
    corpus_obj = KieuCorpus("data\data.txt")
    wset = corpus_obj.sentences()
    print("Unique word: ", len(wset))

    EMBEDDING_DIM = 32
    VOCAB_SIZE = len(corpus_obj.lookup_table()[0])

    net = NPLM(VOCAB_SIZE, EMBEDDING_DIM, 32, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-4)

    word_to_idx, idx_to_word = corpus_obj.lookup_table()
    ngram = corpus_obj.ngram(word_to_idx=word_to_idx)
    ngram = np.array(ngram)
    print(np.max(ngram))

    dataset = torch.utils.data.TensorDataset(
        torch.LongTensor(ngram[:,:-1]),
        torch.LongTensor(ngram[:,-1])
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True
    )

    train(net, criterion, optimizer, dataloader, epoch=40)

    torch.save(net.state_dict(), "weights/nplm.pt")


