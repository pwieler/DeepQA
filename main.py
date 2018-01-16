from __future__ import print_function
from functools import reduce
import re
import numpy as np
import preprocessing.bAbIData as bd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from model.QAModel import QAModel
from utils.utils import time_since
import time


# Train cycle
def train():
    total_loss = 0
    correct = 0

    train_data_size = len(train_loader.dataset)

    train_loss_history = []

    model.train()

    for i, (stories, queries, answers, sl, ql) in enumerate(train_loader, 1):

        stories = Variable(stories.type(torch.LongTensor))
        queries = Variable(queries.type(torch.LongTensor))
        answers = Variable(answers.type(torch.LongTensor))
        sl = Variable(sl.type(torch.LongTensor))
        ql = Variable(ql.type(torch.LongTensor))

        # Sort stories by their length (because of packing in the forward step!)
        sl, perm_idx = sl.sort(0, descending=True)
        stories = stories[perm_idx]
        ql = ql[perm_idx]
        queries = queries[perm_idx]
        answers = answers[perm_idx]

        output = model(stories, queries, sl, ql)

        answers_flat = answers.view(-1)

        loss = criterion(output, answers)

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

    print('Training set: Accuracy: {}/{} ({:.0f}%)'.format(correct, train_data_size, accuracy))

    return train_loss_history, accuracy, total_loss  # loss per epoch


def test():
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

    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, test_data_size, accuracy))

    return test_loss_history, accuracy


class GridSearch():
    def __init__(self):
        self.embeddings = [5, 10, 20, 30, 40, 50]
        self.story_hiddens = [5, 10, 20, 30, 40, 50]
        self.layers = [1, 2, 3, 4]
        self.batch_sizes = [32]
        self.learning_rates = [0.001, 0.0001]
        self.params = []

    def generateParamSet(self):

        self.params = []

        for b in self.batch_sizes:
            for lr in self.learning_rates:
                for l in self.layers:
                    for s in self.story_hiddens:
                        for e in self.embeddings:
                            self.params.append([e, s, l, b, lr])

        return self.params


if __name__ == "__main__":

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
    PRINT_LOSS = True

    ## Load data
    voc = bd.Vocabulary()
    train_instances = []
    test_instances = []

    if BABI_TASK is 0:
        voc.extend_with_file("data/tasks_1-20_v1-2/en/test_data")
        train_instances = bd.BAbIInstance.instances_from_file("data/tasks_1-20_v1-2/en/test_data")
        test_instances = bd.BAbIInstance.instances_from_file("data/tasks_1-20_v1-2/en/test_data")

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

    voc.sort_ids()

    for inst in train_instances:
        inst.vectorize(voc)

    for inst in test_instances:
        inst.vectorize(voc)

    train_dataset = bd.BAbiDataset(train_instances)
    test_dataset = bd.BAbiDataset(test_instances)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    g = GridSearch()
    params = g.generateParamSet()

    for i, param_set in enumerate(params):

        print('\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\nParam-Set: %d of %d' % (i, len(params)))

        ## Parameters
        EMBED_HIDDEN_SIZE = param_set[0]
        STORY_HIDDEN_SIZE = param_set[1]
        QUERY_HIDDEN_SIZE = param_set[0]  # note: since we are adding the encoded query to the embedded stories,
        #  QUERY_HIDDEN_SIZE should be equal to EMBED_HIDDEN_SIZE
        N_LAYERS = param_set[2]
        BATCH_SIZE = param_set[3]
        EPOCHS = 50
        VOC_SIZE = len(voc)
        LEARNING_RATE = param_set[4]

        ## Initialize Model and Optimizer
        model = QAModel(VOC_SIZE, EMBED_HIDDEN_SIZE, STORY_HIDDEN_SIZE, QUERY_HIDDEN_SIZE, VOC_SIZE, N_LAYERS)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.NLLLoss()

        ## Print setting
        # print('Challenge selected: %s' % challenge.split('_')[3])
        print('\nSettings:\nEMBED_HIDDEN_SIZE: %d\nSTORY_HIDDEN_SIZE: %d\nQUERY_HIDDEN_SIZE: %d'
              '\nN_LAYERS: %d\nBATCH_SIZE: %d\nEPOCHS: %d\nVOC_SIZE: %d\nLEARNING_RATE: %f\n' % (
                  EMBED_HIDDEN_SIZE, STORY_HIDDEN_SIZE, QUERY_HIDDEN_SIZE, N_LAYERS, BATCH_SIZE, EPOCHS, VOC_SIZE,
                  LEARNING_RATE))

        # print(model)


        ## Start training
        start = time.time()
        if PRINT_LOSS:
            print("Training for %d epochs..." % EPOCHS)

        train_loss_history = []
        test_loss_history = []

        train_acc_history = []
        test_acc_history = []

        for epoch in range(1, EPOCHS + 1):
            print("Epoche: %d" % epoch)
            # Train cycle
            train_loss, train_accuracy, total_loss = train()

            # Test cycle
            test_loss, test_accuracy = test()

            # Add Loss to history
            train_loss_history = train_loss_history + train_loss
            test_loss_history = test_loss_history + test_loss

            # Add Loss to history
            train_acc_history.append(train_accuracy)  # = train_acc_history + [train_accuracy]
            test_acc_history.append(test_accuracy)  # = test_acc_history + test_accuracy

        # Plot Loss
        if PLOT_LOSS:
            plt.figure()
            plt.plot(train_loss_history, label='train-loss', color='b')
            plt.plot(test_loss_history, label='test-loss', color='r')
            plt.legend()
            plt.show()

            plt.figure()
            plt.plot(train_acc_history, label='train-accuracy', color='b')
            plt.plot(test_acc_history, label='test-accuracy', color='r')
            plt.legend()
            plt.show()
