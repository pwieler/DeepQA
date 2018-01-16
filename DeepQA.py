from __future__ import print_function
from functools import reduce
import re
import numpy as np
import matplotlib.pyplot as plt
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.init as init
import preprocessing.bAbIData as bd


# Some utility functions
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class QADataset(Dataset):
    def __init__(self, story, query, answer, story_lengths, query_lengths):
        self.story = story
        self.query = query
        self.answer = answer
        self.story_lengths = story_lengths
        self.query_lengths = query_lengths
        self.len = len(story)

    def __getitem__(self, index):
        return self.story[index], self.query[index], self.answer[index], self.story_lengths[index], self.query_lengths[
            index]

    def __len__(self):
        return self.len


class QAModel(nn.Module):
    def __init__(self, input_size, embedding_size, story_hidden_size, query_hidden_size, output_size, n_layers=1,
                 bidirectional=False, binary_task=False):
        super(QAModel, self).__init__()

        self.voc_size = input_size
        self.embedding_size = embedding_size
        self.story_hidden_size = story_hidden_size
        self.query_hidden_size = query_hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1
        self.target_size = output_size
        if binary_task:
            self.target_size = 1
        self.story_embedding = nn.Embedding(input_size,
                                            embedding_size)  # Embedding bildet ab von Vokabular (Indize) auf n-dim Raum

        self.story_rnn = nn.GRU(embedding_size, story_hidden_size, n_layers, bidirectional=bidirectional,
                                batch_first=True, dropout=0.3)

        self.query_embedding = nn.Embedding(input_size, embedding_size)
        self.query_rnn = nn.GRU(embedding_size, query_hidden_size, n_layers, bidirectional=bidirectional,
                                batch_first=True, dropout=0.3)

        # info: if we use the old-forward function fc-layer has input-length: "story_hidden_size+query_hidden_size"
        self.fc = nn.Linear(story_hidden_size, self.target_size)
        self.softmax = nn.LogSoftmax()

    # this is the old forward version! below version with question_code performs much better!!
    def old_forward(self, story, query, story_lengths, query_lengths):
        # input shape: B x S (input size)

        # story has dimension batch_size * number of words
        batch_size = story.size(0)

        # Create hidden states for RNNs
        story_hidden = self._init_hidden(batch_size, self.story_hidden_size)
        query_hidden = self._init_hidden(batch_size, self.query_hidden_size)

        # Create Story-Embeddings
        s_e = self.story_embedding(story)  # encodings have size: batch_size*length_of_sequence*EMBBEDDING_SIZE

        # packed Story-Embeddings into RNN
        packed_story = torch.nn.utils.rnn.pack_padded_sequence(s_e, story_lengths.data.cpu().numpy(),
                                                               batch_first=True)  # pack story
        story_output, story_hidden = self.story_rnn(packed_story, story_hidden)
        # unpacking is not necessary, because we use hidden states of RNN

        q_e = self.query_embedding(query)
        query_output, query_hidden = self.query_rnn(q_e, query_hidden)

        merged = torch.cat([story_hidden[0], query_hidden[0]], 1)
        merged = merged.view(batch_size, -1)
        fc_output = self.fc(merged)
        sm_output = self.softmax(fc_output)

        return sm_output

    # new forward-function with question-code
    # achieves 100% on Task 1!!
    # --> question-code is like an attention-mechanism!
    def forward(self, story, query, story_lengths, query_lengths):
        # Calculate Batch-Size
        batch_size = story.size(0)

        # Make a hidden
        story_hidden = self._init_hidden(batch_size, self.story_hidden_size)
        query_hidden = self._init_hidden(batch_size, self.query_hidden_size)

        # Embed query
        q_e = self.query_embedding(query)
        # Encode query-sequence with RNN
        query_output, query_hidden = self.query_rnn(q_e, query_hidden)

        # question_code contains the encoded question!
        # --> we give this directly into the story_rnn,
        # so that the story_rnn can focus on the question already
        # and can forget unnecessary information!
        question_code = query_hidden[0]
        question_code = question_code.view(batch_size, 1, self.query_hidden_size)
        question_code = question_code.repeat(1, story.size(1), 1)

        # Embed story
        s_e = self.story_embedding(story)

        # Combine story-embeddings with question_code
        combined = s_e + question_code

        # put combined tensor into story_rnn --> attention-mechansism through question_code
        packed_story = torch.nn.utils.rnn.pack_padded_sequence(combined, story_lengths.data.cpu().numpy(),
                                                               batch_first=True)  # pack story
        story_output, story_hidden = self.story_rnn(packed_story, story_hidden)
        # remember: because we use the hidden states of the RNN, we don't have to unpack the tensor!

        # Do softmax on the encoded story tensor!
        fc_output = self.fc(story_hidden[0])
        sm_output = self.softmax(fc_output)

        return sm_output

    def _init_hidden(self, batch_size, hidden_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, hidden_size)
        return Variable(hidden)


class QAFFModel(nn.Module):
    # This modell overfits greatly! Not suitable for the problem, but good to illustrate why we use the RNN!
    def __init__(self, input_size, embedding_size, story_hidden_size, query_hidden_size, output_size, n_layers=1,
                 bidirectional=False, s_len=-1, q_len=-1):
        super(QAFFModel, self).__init__()
        assert (s_len > 1)
        assert (q_len > 1)
        self.voc_size = input_size
        self.embedding_size = embedding_size
        self.story_hidden_size = story_hidden_size
        self.query_hidden_size = query_hidden_size
        self.n_layers = n_layers
        self.s_len = s_len
        self.q_len = q_len
        self.story_embedding = nn.Embedding(input_size,
                                            embedding_size)  # Embedding bildet ab von Vokabular (Indize) auf n-dim Raum

        self.query_embedding = nn.Embedding(input_size, embedding_size)

        # info: if we use the old-forward function fc-layer has input-length: "story_hidden_size+query_hidden_size"
        fc1o = int(np.floor(0.5 * embedding_size * (self.q_len + self.s_len)))
        self.fc1 = nn.Linear(embedding_size * (self.q_len + self.s_len), fc1o)
        init.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
        init.constant(self.fc1.bias, 0.1)
        self.fc1a = nn.Tanh()
        self.dropo = nn.Dropout()
        self.fc2 = nn.Linear(fc1o, output_size)
        init.xavier_uniform(self.fc2.weight, gain=np.sqrt(2))
        init.constant(self.fc2.bias, 0.1)
        self.softmax = nn.LogSoftmax()
        # this is the old forward version! below version with question_code performs much better!!

    def forward(self, story, query, story_lengths, query_lengths):
        # input shape: B x S (input size)

        # story has dimension batch_size * number of words
        batch_size = story.size(0)

        # Create Story-Embeddings
        s_e = self.story_embedding(story)  # encodings have size: batch_size*length_of_sequence*EMBBEDDING_SIZE

        # Create Question embedding
        q_e = self.query_embedding(query)

        # Transform the tensors to do the processing
        s_e = s_e.view(batch_size, -1)
        q_e = q_e.view(batch_size, -1)
        merged = torch.cat([s_e, q_e], 1)

        # First fc with tanh
        fc_output = self.fc1(merged)
        th_out = self.fc1a(fc_output)

        # Apply dropout
        th_out1 = self.dropo(th_out)
        out = self.fc2(th_out1)
        sm_output = self.softmax(out)

        return sm_output


# Train cycle
def train(model, train_loader, criterion, optimizer, epoch, start, PRINT_LOSS=True):
    total_loss = 0
    correct = 0

    train_data_size = len(train_loader.dataset)

    train_loss_history = []

    model.train()

    for i, (stories_bd, queries_bd, answers_bd, sl_bd, ql_bd) in enumerate(train_loader, 1):

        stories = Variable(stories_bd.type(torch.LongTensor))
        queries = Variable(queries_bd.type(torch.LongTensor))
        answers = Variable(answers_bd.type(torch.LongTensor))
        sl = Variable(sl_bd.type(torch.LongTensor))
        ql = Variable(ql_bd.type(torch.LongTensor))

        # Sort stories by their length (because of packing in the forward step!)
        sl, perm_idx = sl.sort(0, descending=True)
        stories = stories[perm_idx]
        ql = ql[perm_idx]
        queries = queries[perm_idx]
        answers = answers[perm_idx]

        output = model(stories, queries, sl, ql)

        loss = criterion(output, answers.view(-1))

        total_loss += loss.data[0]

        train_loss_history.append(loss.data[0])

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if PRINT_LOSS:
            if i % 1 == 0:
                print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(time_since(start), epoch,
                                                                                    i * len(stories),
                                                                                    len(train_loader.dataset),
                                                                                    100. * i * len(stories) / len(
                                                                                        train_loader.dataset),
                                                                                    loss.data[0]))

        pred_answers = output.data.max(1)[1]
        correct += pred_answers.eq(
                answers.data.view_as(pred_answers)).cpu().sum()  # calculate how many labels are correct

    accuracy = 100. * correct / train_data_size

    # if PRINT_LOSS:
    print('Training set: Accuracy: {}/{} ({:.0f}%)'.format(correct, train_data_size, accuracy))

    return train_loss_history, accuracy, total_loss  # loss per epoch


def test(model, test_loader, criterion, PRINT_LOSS):
    model.eval()

    if PRINT_LOSS:
        print("evaluating trained model ...")

    correct = 0
    test_data_size = len(test_loader.dataset)

    test_loss_history = []

    for stories, queries, answers, sl, ql in test_loader:
        stories = Variable(stories.type(torch.LongTensor))
        queries = Variable(queries.type(torch.LongTensor))
        answers = Variable(answers.type(torch.LongTensor))
        sl = Variable(sl.type(torch.LongTensor))
        ql = Variable(ql.type(torch.LongTensor))

        # Sort stories by their length
        sl, perm_idx = sl.sort(0, descending=True)
        stories = stories[perm_idx]
        # ql, perm_idx = ql.sort(0, descending=True) # if we sort query also --> then they do not fit together!
        ql = ql[perm_idx]
        queries = queries[perm_idx]
        answers = answers[perm_idx]

        output = model(stories, queries, sl, ql)

        loss = criterion(output, answers.view(-1))
        test_loss_history.append(loss.data[0])

        pred_answers = output.data.max(1)[1]
        correct += pred_answers.eq(
                answers.data.view_as(pred_answers)).cpu().sum()  # calculate how many labels are correct

    accuracy = 100. * correct / test_data_size

    # if PRINT_LOSS:
    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, test_data_size, accuracy))

    return test_loss_history, accuracy


def main():
    BABI_TASK = 1

    ## Parameters
    EMBED_HIDDEN_SIZE = 50
    STORY_HIDDEN_SIZE = 50
    QUERY_HIDDEN_SIZE = 50  # note: since we are adding the encoded query to the embedded stories,
    #  QUERY_HIDDEN_SIZE should be equal to EMBED_HIDDEN_SIZE

    N_LAYERS = 1
    BATCH_SIZE = 32
    EPOCHS = 40
    LEARNING_RATE = 0.001  # 0.0001

    PLOT_LOSS = True

    ## Load data
    voc = bd.Vocabulary()
    train_instances = []
    test_instances = []

    if BABI_TASK is 1:
        voc.extend_with_file("data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt")
        train_instances = bd.BAbIInstance.instances_from_file(
                "data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt")
        test_instances = bd.BAbIInstance.instances_from_file(
                "data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_test.txt")

    elif BABI_TASK is 2:
        voc.extend_with_file("data/tasks_1-20_v1-2/en/qa2_two-supporting-facts_train.txt")
        train_instances = bd.BAbIInstance.instances_from_file(
                "data/tasks_1-20_v1-2/en/qa2_two-supporting-facts_train.txt")
        test_instances = bd.BAbIInstance.instances_from_file(
                "data/tasks_1-20_v1-2/en/qa2_two-supporting-facts_test.txt")

    elif BABI_TASK is 3:
        voc.extend_with_file("data/tasks_1-20_v1-2/en/qa3_three-supporting-facts_train.txt")
        train_instances = bd.BAbIInstance.instances_from_file(
                "data/tasks_1-20_v1-2/en/qa3_three-supporting-facts_train.txt")
        test_instances = bd.BAbIInstance.instances_from_file(
                "data/tasks_1-20_v1-2/en/qa3_three-supporting-facts_test.txt")
    elif BABI_TASK is 6:
        voc.extend_with_file("data/tasks_1-20_v1-2/en/qa6_yes-no-questions_train.txt")
        train_instances = bd.BAbIInstance.instances_from_file("data/tasks_1-20_v1-2/en/qa6_yes-no-questions_train.txt")
        test_instances = bd.BAbIInstance.instances_from_file("data/tasks_1-20_v1-2/en/qa6_yes-no-questions_test.txt")

    for inst in train_instances:
        inst.vectorize(voc)

    for inst in test_instances:
        inst.vectorize(voc)

    train_dataset = bd.BAbiDataset(train_instances)
    test_dataset = bd.BAbiDataset(test_instances)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    ## Initialize Model and Optimizer

    model = QAModel(len(voc), EMBED_HIDDEN_SIZE, STORY_HIDDEN_SIZE, QUERY_HIDDEN_SIZE, len(voc), N_LAYERS)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(model)

    ## Start training

    start = time.time()
    print("Training for %d epochs..." % EPOCHS)

    train_loss_history = []
    test_loss_history = []

    train_acc_history = []
    test_acc_history = []

    for epoch in range(1, EPOCHS + 1):
        # Train cycle
        train_loss, train_accuracy, total_loss = train(model, train_loader, criterion, optimizer, epoch, start)

        # Test cycle
        test_loss, test_accuracy = test(model, test_loader, criterion)

        # Add Loss to history
        train_loss_history = train_loss_history + train_loss
        test_loss_history = test_loss_history + test_loss

        # Add Loss to history
        train_acc_history.append(train_accuracy)  # = train_acc_history + [train_accuracy]
        test_acc_history.append(test_accuracy)  # = test_acc_history + test_accuracy

    # Plot Loss
    if PLOT_LOSS:
        plt.figure()
        plt.hold(True)
        plt.plot(train_loss_history, 'b')
        plt.plot(test_loss_history, 'r')
        plt.show()

        plt.figure()
        plt.hold(True)
        plt.plot(train_acc_history, 'b')
        plt.plot(test_acc_history, 'r')
        plt.show()


if __name__ == "__main__":
    main()
