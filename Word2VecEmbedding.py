import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.functional as functional
from random import randint
from typing import List
import re
import os
import pickle
import bAbIData as bd
import matplotlib.pyplot as plt

print("PyTorch Version: " + torch.__version__)


class W2VInstance:
    def __init__(self, word, context):
        self.word = word
        self.context = context

    def get_random_context(self):
        return self.context[randint(0, len(self.context) - 1)]

    @staticmethod
    def instances_from_file(path, context_l=1, context_r=1):
        instances = []

        with open(path, 'r') as f:
            for line in f.readlines():
                instances += W2VInstance.instances_from_line(line, context_l, context_r)

        return instances

    @staticmethod
    def instances_from_line(line, context_l=1, context_r=1):
        instances = []
        words = re.split('(\W)', line)

        words = [word for word in words if word not in ["", " ", "\n", "\r"] and not word.isnumeric()]

        for i in range(len(words)):
            word = words[i]
            context = []

            for j in range(max(0, i - context_l), min(len(words), i + context_r)):
                if j is not i:
                    context += words[j]

            if len(context) is 0:
                continue

            instances.append(W2VInstance(word, context))

        return instances


class W2VSkipGramEmbedding(nn.Module):
    """This Module is used for training a PyTorch nn.embedding with a word2vec skip gram approach."""

    def __init__(self, embeddings: nn.Embedding):
        """
        Hand over an embedding which will be trained as first half of a word2vec skip gram network.
        (One-Hot-Input->embedding matrix->hidden layer (feature vector)->softmax vs target label)

        :param embeddings: Embeddings is basically a linear layer from the One-Hot-Input of dimension
        |Vocabulary|=VxN=|Features of feature vector|. embeddings.weight.data is the underlying matrix corresponding
        to the linear layer.
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
        hidden_embedding = self.embeddings(input).view(1, -1)

        # Calculate values for context probabilities from feature vector
        out = self.fc(hidden_embedding)

        # Convert output to probabilities with logarithmic softmax.
        out = functional.log_softmax(out)

        return out


def train(network: W2VSkipGramEmbedding, instances: List[W2VInstance], voc: bd.Vocabulary, number_epochs=1,
          plot_loss=False):
    loss_calc = nn.NLLLoss()

    optimizer = optim.SGD(network.parameters(), lr=0.001)

    plt.ion()

    loss_history = []

    for epoch in range(number_epochs):
        epoch_loss = 0

        for instance in instances:
            # Using a random context word for training
            input_id = voc.word_to_id(instance.word)
            target_id = voc.word_to_id(instance.get_random_context())

            network_input = autograd.Variable(torch.LongTensor([input_id]))
            network_target = autograd.Variable(torch.LongTensor([target_id]))

            context_probabilities = network.forward(network_input)

            loss = loss_calc.forward(context_probabilities, network_target)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.data[0]

        loss_instance = epoch_loss / len(instances)
        loss_history.append(loss_instance)

        print("Loss per instance of Epoch " + str(epoch) + ": " + str(loss_instance))
        plt.plot(loss_history)
        plt.pause(0.001)


def main():
    load_predefined = False
    learning_file = "data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_test.txt"

    if load_predefined:
        voc = bd.Vocabulary()

        if os.path.isfile("embedding.tensor") and os.path.isfile("vocabulary.pickle"):
            voc.embedding.weight.data = torch.load("embedding.tensor")
            voc.voc_dict = pickle.load(open("vocabulary.pickle", "r"))
            print("Loaded previously defined vocabulary and embedding.")
        else:
            print("could not find  files to load.")
            exit(1)
    else:
        voc = bd.Vocabulary(learning_file)
        voc.initialize_embedding()

    w2v_instances = W2VInstance.instances_from_file(learning_file)

    print("Number of training instances: " + str(len(w2v_instances)))

    w2v_net = W2VSkipGramEmbedding(voc.embedding)

    train(w2v_net, w2v_instances, voc, number_epochs=50, plot_loss=True)

    torch.save(voc.embedding.weight.data, "embedding.tensor")

    pickle.dump(voc.voc_dict, open("vocabulary.pickle", "wb"))

    voc.embedding = w2v_net.embeddings

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    vocs = list(voc.voc_dict.keys())
    for word in vocs[:50]:
        word_em = voc.word_to_tensor(word)
        distances = [(other, cos(voc.word_to_tensor(word), voc.word_to_tensor(other)).data[0]) for other in
                     voc.voc_dict.keys()]

        distances = [(tup[0], round(tup[1], 2)) for tup in distances]
        distances = sorted(distances, key=lambda tup: tup[1], reverse=True)

        print(word + " : " + str(distances))


if __name__ == '__main__':
    main()
