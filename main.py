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
import os


def main():
    # Can be either 1,2,3 or 6 respective to the evaluated task.
    BABI_TASK = 1

    babi_voc_path = {
        0: "data/tasks_1-20_v1-2/en/test_data",
        1: "data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt",
        2: "data/tasks_1-20_v1-2/en/qa2_two-supporting-facts_train.txt",
        3: "data/tasks_1-20_v1-2/en/qa3_three-supporting-facts_train.txt",
        6: "data/tasks_1-20_v1-2/en/qa6_yes-no-questions_train.txt"
        }

    babi_train_path = {
        0: "data/tasks_1-20_v1-2/en/test_data",
        1: "data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt",
        2: "data/tasks_1-20_v1-2/en/qa2_two-supporting-facts_train.txt",
        3: "data/tasks_1-20_v1-2/en/qa3_three-supporting-facts_train.txt",
        6: "data/tasks_1-20_v1-2/en/qa6_yes-no-questions_train.txt"
        }

    babi_test_path = {
        0: "data/tasks_1-20_v1-2/en/test_data",
        1: "data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_test.txt",
        2: "data/tasks_1-20_v1-2/en/qa2_two-supporting-facts_test.txt",
        3: "data/tasks_1-20_v1-2/en/qa3_three-supporting-facts_test.txt",
        6: "data/tasks_1-20_v1-2/en/qa6_yes-no-questions_test.txt"
        }

    PREVIOUSLY_TRAINED_MODEL = None
    ONLY_EVALUATE = False

    ## GridSearch Parameters
    EPOCHS = [40]  # Mostly you only want on epoch param, unless you want equal models with different training times.
    EMBED_HIDDEN_SIZES = [50]
    STORY_HIDDEN_SIZE = [100]
    N_LAYERS = [1, 2]
    BATCH_SIZE = [100]
    LEARNING_RATE = [0.001]  # 0.0001

    ## Output parameters
    # Makes the training halt between every param set until you close the plot windows. Plots are saved either way.
    PLOT_LOSS_INTERACTIVE = False
    PRINT_BATCHWISE_LOSS = False

    grid_search_params = GridSearchParamDict(EMBED_HIDDEN_SIZES, STORY_HIDDEN_SIZE, N_LAYERS, BATCH_SIZE, LEARNING_RATE,
                                             EPOCHS)

    voc, train_instances, test_instances = load_data(babi_voc_path[BABI_TASK], babi_train_path[BABI_TASK],
                                                     babi_test_path[BABI_TASK])

    # Converts the words of the instances from string representation to integer representation using the vocabulary.
    vectorize_data(voc, train_instances, test_instances)

    for i, param_dict in enumerate(grid_search_params):
        print('\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\nParam-Set: %d of %d' % (i + 1, len(grid_search_params)))

        embedding_size = param_dict["embedding_size"]
        story_hidden_size = param_dict["story_hidden_size"]
        query_hidden_size = story_hidden_size
        n_layers = param_dict["layers"]
        learning_rate = param_dict["learning_rate"]
        batch_size = param_dict["batch_size"]
        epochs = param_dict["epochs"]
        voc_len = len(voc)

        ## Print setting
        readable_params = '\nSettings:\nEMBED_HIDDEN_SIZE: %d\nSTORY_HIDDEN_SIZE: %d\nQUERY_HIDDEN_SIZE: %d' \
                          '\nN_LAYERS: %d\nBATCH_SIZE: %d\nEPOCHS: %d\nVOC_SIZE: %d\nLEARNING_RATE: %f\n' % (
                              embedding_size, story_hidden_size, query_hidden_size, n_layers, batch_size, epochs,
                              voc_len, learning_rate)

        print(readable_params)

        train_loader, test_loader = prepare_dataloaders(train_instances, test_instances, batch_size)

        ## Initialize Model and Optimizer
        model = QAModel(voc_len, embedding_size, story_hidden_size, voc_len, n_layers)

        # If a path to a state dict of a previously trained model is given, the state will be loaded here.
        if PREVIOUSLY_TRAINED_MODEL is not None:
            model.load_state_dict(torch.load(PREVIOUSLY_TRAINED_MODEL))

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()

        train_loss, test_loss, train_acc, test_acc = conduct_training(model, train_loader, test_loader, optimizer,
                                                                      criterion, only_evaluate=ONLY_EVALUATE,
                                                                      print_loss=PRINT_BATCHWISE_LOSS, epochs=epochs)

        params = [embedding_size, story_hidden_size, query_hidden_size, n_layers, batch_size, epochs, voc_len,
                  learning_rate, epochs]
        log(BABI_TASK, train_loss, test_loss, params, train_acc, test_acc, readable_params, model)

        # Plot Loss
        if PLOT_LOSS_INTERACTIVE:
            plot_data_in_window(train_loss, test_loss, train_acc, test_acc)


def train(model, train_loader, optimizer, criterion, start, epoch, print_loss=False):
    total_loss = 0
    correct = 0
    train_loss_history = []

    train_data_size = len(train_loader.dataset)

    # Set model in training mode
    model.train()

    # The train loader will give us batches of data according to batch size. Example:
    # Batch size is 32 training samples and stories are padded to 66 words (each represented by an integer for the
    # vocabulary index)
    # The stories parameter will contain a tensor of size 32x66. Likewise for the other parameters
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

        if print_loss:
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


def test(model, test_loader, criterion, PRINT_LOSS=False):
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


def prepare_dataloaders(train_instances, test_instances, batch_size, shuffle=True):
    train_dataset = bd.BAbiDataset(train_instances)
    test_dataset = bd.BAbiDataset(test_instances)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


class GridSearchParamDict():
    def __init__(self, embeddings, story_hidden_sizes, layers, batch_sizes, learning_rates, epochs):
        self.embeddings = embeddings
        self.story_hiddens = story_hidden_sizes
        self.layers = layers
        self.batch_sizes = batch_sizes
        self.learning_rates = learning_rates
        self.epochs = epochs

        self.params = self.generate_param_set()

    def __len__(self):
        return len(self.params)

    def __getitem__(self, key):
        return self.params[key]

    def generate_param_set(self):
        self.params = []

        for b in self.batch_sizes:
            for lr in self.learning_rates:
                for l in self.layers:
                    for s in self.story_hiddens:
                        for em in self.embeddings:
                            for ep in self.epochs:
                                self.params.append({
                                    "embedding_size":    em,
                                    "story_hidden_size": s,
                                    "layers":            l,
                                    "batch_size":        b,
                                    "learning_rate":     lr,
                                    "epochs":            ep
                                    })

        return self.params


def load_data(voc_path, train_path, test_path):
    voc = bd.Vocabulary()
    train_instances = []
    test_instances = []

    voc.extend_with_file(voc_path)
    train_instances = bd.BAbIInstance.instances_from_file(train_path)
    test_instances = bd.BAbIInstance.instances_from_file(test_path)

    voc.sort_ids()

    return voc, train_instances, test_instances


def vectorize_data(voc, train_instances, test_instances):
    # At this point, training instances have been loaded with real word sentences.
    # Using the vocabulary we convert the words into integer representations, so they can converted with an embedding
    # later on.
    for inst in train_instances:
        inst.vectorize(voc)

    for inst in test_instances:
        inst.vectorize(voc)


def conduct_training(model, train_loader, test_loader, optimizer, criterion, only_evaluate=False, print_loss=False,
                     epochs=1):
    train_loss_history = []
    test_loss_history = []

    train_acc_history = []
    test_acc_history = []

    ## Start training
    start = time.time()
    if print_loss:
        print("Training for %d epochs..." % epochs)

    for epoch in range(1, epochs + 1):
        print("Epoche: %d" % epoch)
        # Train cycle
        if not only_evaluate:
            train_loss, train_accuracy, total_loss = train(model, train_loader, optimizer, criterion, start, epoch,
                                                           print_loss)

        # Test cycle
        test_loss, test_accuracy = test(model, test_loader, criterion, PRINT_LOSS=False)

        # Add Loss to history
        if not only_evaluate:
            train_loss_history = train_loss_history + train_loss
        test_loss_history = test_loss_history + test_loss

        # Add Loss to history
        if not only_evaluate:
            train_acc_history.append(train_accuracy)
        test_acc_history.append(test_accuracy)

    return train_loss_history, test_loss_history, train_acc_history, test_acc_history


def plot_data_in_window(train_loss, test_loss, train_acc, test_acc):
    plt.figure()
    plt.plot(train_loss, label='train-loss', color='b')
    plt.plot(test_loss, label='test-loss', color='r')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_acc, label='train-accuracy', color='b')
    plt.plot(test_acc, label='test-accuracy', color='r')
    plt.legend()
    plt.show()  # Train cycle


def concatenated_params(params):
    params_str = [str(x) for x in params]
    params_str = reduce(lambda x, y: x + '_' + y, params_str)

    return params_str


def log(task, train_loss, test_loss, params, train_accuracy, test_accuracy, params_file, model, plots=True):
    param_str = concatenated_params(params)

    date = str(time.strftime("%Y:%m:%d:%H:%M:%S"))
    fname = "results/" + date.replace(":", "_") + "_" + param_str + "_task_" + str(task) + "/"
    try:
        os.stat(fname)
    except:
        os.mkdir(fname)
    tr_loss = np.array(train_loss)
    te_loss = np.array(test_loss)
    tr_acc = np.array(train_accuracy)
    te_acc = np.array(test_accuracy)
    tr_loss.tofile(fname + "train_loss.csv", sep=";")
    te_loss.tofile(fname + "test_loss.csv", sep=";")
    tr_acc.tofile(fname + "train_accuracy.csv", sep=";")
    te_acc.tofile(fname + "test_accuracy.csv", sep=";")
    if plots == True:
        plt.figure()
        plt.plot(train_loss, label='train-loss', color='b')
        plt.plot(test_loss, label='test-loss', color='r')
        plt.legend()
        plt.savefig(fname + "loss_history.png")
        plt.figure()
        plt.plot(train_accuracy, label='train-accuracy', color='b')
        plt.plot(test_accuracy, label='test-accuracy', color='r')
        plt.legend()
        plt.savefig(fname + "acc_history.png")
        plt.close("all")
    with open(fname + "params.txt", "w") as text_file:
        text_file.write(params_file)
    torch.save(model.state_dict(), fname + "trained_model.pth")


if __name__ == "__main__":
    main()
