import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as functional
import math
import re

from random import randint

vocabulary = dict()
word_context = []


class W2VSkipGramEmbedding(nn.Module):
    """This Module is used for training a PyTorch nn.embedding with a word2vec skip gram approach."""

    def __init__(self, embeddings: nn.Embedding):
        """
        Hand over an embedding which will be trained as first half of a word2vec skip gram network.
        (One-Hot-Input->embedding matrix->hidden layer (feature vector)->softmax vs target label)

        :param embeddings: Embeddings is basically a linear layer from the One-Hot-Input of dimension |Vocabulary|=VxN=|Features of feature vector|. embeddings.weight.data is the underlying matrix corresponding to the linear layer.
        """
        super(W2VSkipGramEmbedding, self).__init__()

        # Linear Layer VxN
        self.embeddings = embeddings

        # Linear Layer NxV
        self.fc = nn.Linear(self.embeddings.embedding_dim, self.embeddings.num_embeddings)

    def forward(self, input) -> torch.FloatTensor:
        """
        Make forward pass through the network and calculate result.

        :param input: A one hot Tensor representing the input word.
        :return: An output Tensor representing the probabilites of each word to be in context (One hot representation)
        """

        # Get the feature vector
        hidden_embedding = self.embeddings(input)

        # Calculate values for context probabilities from feature vector
        out = self.fc(hidden_embedding)

        # Convert output to probabilities with logarithmic softmax.
        out = functional.log_softmax(out)

        return out


def word_to_index(word):
    return vocabulary[word]


def choose_target_randomly(context_idxs):
    ids = [word_id for word_id in context_idxs if word_id is not None]

    if len(ids) == 1:
        return ids[0]

    return ids[randint(0, len(ids) - 1)]


def train(network: W2VSkipGramEmbedding, optimizer: optim.Optimizer, number_epochs=10):
    loss_calc = nn.NLLLoss()

    print("Size of vocabulary: " + str(len(vocabulary)))
    print("Size of context data: " + str(len(word_context)))

    for epoch in range(number_epochs):
        epoch_loss = 0

        for (word_idx, context_idxs) in word_context:
            # Using a random context word for training
            network_input = autograd.Variable(torch.LongTensor([word_idx]))
            network_target = autograd.Variable(torch.LongTensor([choose_target_randomly(context_idxs)]))

            network.zero_grad()

            context_probabilities = network.forward(network_input)

            loss = loss_calc.forward(context_probabilities, network_target)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.data

        print("Loss of Epoch " + str(epoch) + ": " + str(epoch_loss[0]))


def add_to_vocabulary(text: str):
    word_set = set()

    for word_blob in text.split():
        alnum_word = "".join([c for c in word_blob if c.isalnum()])
        symbols = "".join([c for c in word_blob if not c.isalnum()])

        if len(alnum_word) > 0:
            if alnum_word not in word_set: word_set.add(alnum_word)

        if len(symbols) > 0:
            if symbols not in word_set: word_set.add(symbols)

    for word in word_set:
        if word not in vocabulary: vocabulary[word] = len(vocabulary)


def word_to_id(word):
    # This is a trick entry for words that are non existent.
    if word is None or word not in vocabulary:
        return len(vocabulary)

    return vocabulary[word]


def add_to_data(text: str):
    words = re.split('(\W)', text)

    words = [word for word in words if word not in ["", " ", "\n", "\r"] and not word.isnumeric()]

    for i in range(len(words)):
        w2p = words[i - 2] if i - 2 > 0 else None
        w1p = words[i - 1] if i - 1 > 0 else None

        w1a = words[i + 1] if i + 1 < len(words) else None
        w2a = words[i + 2] if i + 2 < len(words) else None

        if w2p==None and w1p==None and w1a==None and w2a==None:
            continue

        word_id = word_to_id(words[i])
        context = (word_to_id(w2p), word_to_id(w1p), word_to_id(w1a), word_to_id(w2a))

        # Only add entry if word is in vocabulary
        if word_id != len(vocabulary):
            word_context.append((word_id, context))


def file_to_vocabulary(path="data/qa2_two-supporting-facts_train.txt"):
    with open(path, 'r') as f:
        for line in f.readlines():
            add_to_vocabulary(line)


def file_to_data(path="data/qa2_two-supporting-facts_train.txt"):
    with open(path, 'r') as f:
        for line in f.readlines():
            add_to_data(line)


def main():
    file_to_vocabulary()

    file_to_data()

    # We want to represent a vocabulary with v one hot vectors by a feature vector with n features.
    # Add +1 to each to represent words that are not in the vocabulary.
    v = len(vocabulary) + 10
    n = math.log2(len(vocabulary)) + 1

    our_custom_embedding = nn.Embedding(v, int(n))

    w2v_net = W2VSkipGramEmbedding(our_custom_embedding)

    optimizer = optim.SGD(w2v_net.parameters(), lr=0.01)

    train(w2v_net, optimizer)


if __name__ == '__main__':
    main()
