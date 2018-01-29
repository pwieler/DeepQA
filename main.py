from __future__ import print_function
from functools import reduce
import pickle
import numpy as np
import preprocessing.bAbIData as bd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model.QAModel import QAModel
from utils.utils import time_since
import time
import os


def main():
    start = time.time()

    use_cuda = True

    if use_cuda:
        print("Cuda available: " + str(torch.cuda.is_available()))

    # Can be one or multiple of 1,2,3 or 6 respective to the evaluated tasks.
    BABI_TASKS = [1,2,3,6]

    print('Training for tasks:' + "".join([" QA" + str(t) for t in BABI_TASKS]))

    base_path = "data/tasks_1-20_v1-2/en"

    babi_voc_path = {
        0: "data/tasks_1-20_v1-2/en/test_data",
        1: base_path + "/" + "qa1_single-supporting-fact_train.txt",
        2: base_path + "/" + "qa2_two-supporting-facts_train.txt",
        3: base_path + "/" + "qa3_three-supporting-facts_train.txt",
        6: base_path + "/" + "qa6_yes-no-questions_train.txt"
        }

    babi_train_path = {
        0: "data/tasks_1-20_v1-2/en/test_data",
        1: base_path + "/" + "qa1_single-supporting-fact_train.txt",
        2: base_path + "/" + "qa2_two-supporting-facts_train.txt",
        3: base_path + "/" + "qa3_three-supporting-facts_train.txt",
        6: base_path + "/" + "qa6_yes-no-questions_train.txt"
        }

    babi_test_path = {
        0: "data/tasks_1-20_v1-2/en/test_data",
        1: base_path + "/" + "qa1_single-supporting-fact_test.txt",
        2: base_path + "/" + "qa2_two-supporting-facts_test.txt",
        3: base_path + "/" + "qa3_three-supporting-facts_test.txt",
        6: base_path + "/" + "qa6_yes-no-questions_test.txt"
        }

    PREVIOUSLY_TRAINED_PATH = None
    MODEL_NAME = "trained_model.pth"
    VOC_FILE_NAME = "vocabulary.pkl"

    ONLY_EVALUATE = False

    ## GridSearch Parameters
    EPOCHS = [200]  # Mostly you only want one epoch param, unless you want equal models with different training times.
    EMBED_HIDDEN_SIZES = [40]
    STORY_HIDDEN_SIZE = [300]
    N_LAYERS = [3]
    BATCH_SIZE = [256]
    LEARNING_RATE = [0.001]  # 0.0001

    ## Output parameters
    # Makes the training halt between every param set until you close the plot windows. Plots are saved either way.
    PLOT_LOSS_INTERACTIVE = False
    PRINT_BATCHWISE_LOSS = False

    # Builds up every possible combination of given hyper parameters
    grid_search_params = GridSearchParamDict(EMBED_HIDDEN_SIZES, STORY_HIDDEN_SIZE, N_LAYERS, BATCH_SIZE, LEARNING_RATE,
                                             EPOCHS)

    # The voc is used to translate from a word to a one hot representation or actually an index in an embedding.
    # train_instances and test_instances are self explaining.
    voc, train_instances, test_instances = load_data([babi_voc_path[t] for t in BABI_TASKS],
                                                     [babi_train_path[t] for t in BABI_TASKS],
                                                     [babi_test_path[t] for t in BABI_TASKS])

    if PREVIOUSLY_TRAINED_PATH is not None:
        with open(PREVIOUSLY_TRAINED_PATH + VOC_FILE_NAME, 'rb') as f:
            voc.voc_dict = pickle.load(f)

    # Converts the words of the instances from string representation to integer representation using the vocabulary.
    vectorize_data(voc, train_instances, test_instances)

    for i, param_dict in enumerate(grid_search_params):
        print('\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\nParam-Set: %d of %d' % (i + 1, len(grid_search_params)))

        embedding_size = param_dict["embedding_size"]
        story_hidden_size = param_dict["story_hidden_size"]
        n_layers = param_dict["layers"]
        learning_rate = param_dict["learning_rate"]
        batch_size = param_dict["batch_size"]
        epochs = param_dict["epochs"]
        voc_len = len(voc)

        ## Print setting
        readable_params = '\nSettings:\nEMBED_HIDDEN_SIZE: %d\nSTORY_HIDDEN_SIZE: %d\nN_LAYERS: %d\nBATCH_SIZE: ' \
                          '%d\nEPOCHS: %d\nVOC_SIZE: %d\nLEARNING_RATE: %f\n' % (
                              embedding_size, story_hidden_size, n_layers, batch_size, epochs, voc_len, learning_rate)

        print(readable_params)
        
        print("Number of train instances: {}".format(len(train_instances)))
        print("Number of test instances: {}\n".format(len(test_instances)))

        train_loader, test_loader = prepare_dataloaders(train_instances, test_instances, batch_size)

        ## Initialize Model and Optimizer
        model = QAModel(voc_len, embedding_size, story_hidden_size, voc_len, n_layers, use_cuda=use_cuda)

        # If a path to a state dict of a previously trained model is given, the state will be loaded here.
        if PREVIOUSLY_TRAINED_PATH is not None:
            model.load_state_dict(torch.load(PREVIOUSLY_TRAINED_PATH + MODEL_NAME))

        if use_cuda and torch.cuda.is_available():
            model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()

        train_loss, test_loss, train_acc, test_acc = conduct_training(model, train_loader, test_loader, optimizer,
                                                                      criterion, only_evaluate=ONLY_EVALUATE,
                                                                      print_loss_batchwise=PRINT_BATCHWISE_LOSS,
                                                                      epochs=epochs, use_cuda=use_cuda)

        params = [embedding_size, story_hidden_size, n_layers, batch_size, epochs, voc_len, learning_rate, epochs]
        save_results(BABI_TASKS, train_loss, test_loss, params, train_acc, test_acc, readable_params, model, voc,
                     param_dict)

        # Plot Loss
        if PLOT_LOSS_INTERACTIVE:
            plot_data_in_window(train_loss, test_loss, train_acc, test_acc)


def conduct_training(model, train_loader, test_loader, optimizer, criterion, only_evaluate=False,
                     print_loss_batchwise=False, epochs=1, use_cuda=True):
    """
    Trains a model for multiple epochs and keeps track of the loss and accuracy for each epoch.

    :param model: The nn.module implementation that is to be trained
    :param train_loader: DataLoader for the train data.
    :param test_loader: DataLoader for the test data.
    :param optimizer: Strategy for adapting the parameters with gradients. It is presumed that the optimizer already
            accesses the models parameters.
    :param criterion: Criteria to determine loss.
    :param only_evaluate: If this is true, not training is performed. (For testing pretrained networks)
    :param print_loss_batchwise: Print loss for every batch instead of only for epoch.
    :param epochs: Number of epochs of training.
    :return: Batchwise history of loss in training and testing as well as accuracy.
    """
    train_loss_history = []
    test_loss_history = []

    train_acc_history = []
    test_acc_history = []

    ## Start training
    start = time.time()
    if print_loss_batchwise:
        print("Training for %d epochs..." % epochs)

    for epoch in range(1, epochs + 1):

        print("[{}] Epoche: {}".format(time_since(start), epoch))
        # Train cycle
        if not only_evaluate:
            train_loss, train_accuracy, total_loss = train(model, train_loader, optimizer, criterion, start, epoch,
                                                           print_loss_batchwise, use_cuda=use_cuda)

        # Test cycle
        test_loss, test_accuracy = test(model, test_loader, criterion, print_loss_batchwise=False, use_cuda=use_cuda)

        # Add Loss to history
        if not only_evaluate:
            train_loss_history = train_loss_history + train_loss
        test_loss_history = test_loss_history + test_loss

        # Add Loss to history
        if not only_evaluate:
            train_acc_history.append(train_accuracy)
        test_acc_history.append(test_accuracy)

    return train_loss_history, test_loss_history, train_acc_history, test_acc_history


def train(model, train_loader, optimizer, criterion, start, epoch, print_loss_batchwise=False, use_cuda=True):
    """
    Implements one epoch of model training.

    :param model: The nn.module implementation that is to be trained
    :param train_loader: DataLoader for the train data.
    :param optimizer: Strategy for adapting the parameters with gradients. It is presumed that the optimizer already
            accesses the models parameters.
    :param criterion: Criteria to determine loss.
    :param start: Point of time to measure progress against.
    :param epoch: Identifies the current epoch of training.
    :param print_loss_batchwise: Print loss for every batch.
    :return: Batchwise history of loss in training.
    """
    epoch_start = time.time()

    total_loss = 0
    correct = 0
    train_loss_history = []
    max_batch_repetitions = 0

    train_data_size = len(train_loader.dataset)

    # Set model in training mode
    model.train()

    # The train loader will give us batches of data according to batch size. Example:
    # Batch size is 32 training samples and stories are padded to 66 words (each represented by an integer for the
    # vocabulary index)
    # The stories parameter will contain a tensor of size 32x66. Likewise for the other parameters
    for i, (stories, queries, answers, sl, ql) in enumerate(train_loader, 1):

        j = 0

        repeat_training = True


        stories = create_variable(stories.type(torch.LongTensor), use_cuda)
        queries = create_variable(queries.type(torch.LongTensor), use_cuda)
        answers = create_variable(answers.type(torch.LongTensor), use_cuda)
        sl = create_variable(sl.type(torch.LongTensor), use_cuda)
        ql = create_variable(ql.type(torch.LongTensor), use_cuda)


        while repeat_training and j < 15:
            output = model(stories, queries, sl, ql)

            answers_flat = answers.view(-1)

            loss = criterion(output, answers)

            total_loss += loss.data[0]

            model.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.data[0]
            j += 1

            # Update again if loss is especially high
            if len(train_loss_history) > 4:
                repeat_training = loss_val > 1.0025 * sum(train_loss_history) / len(train_loss_history)
            else:
                repeat_training = False

        if j > max_batch_repetitions:
            max_batch_repetitions = j

        # Calculating elementwise loss per batch
        train_loss_history.append(loss.data[0])

        if print_loss_batchwise:
            if i % 1 == 0:
                print('    [~{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(time_since(epoch_start), epoch,
                                                                                         i * len(stories),
                                                                                         len(train_loader.dataset),
                                                                                         100. * i * len(stories) / len(
                                                                                                 train_loader.dataset),
                                                                                         loss.data[0]))

        pred_answers = output.data.max(1)[1]
        correct += pred_answers.eq(
                answers.data.view_as(pred_answers)).cpu().sum()  # calculate how many labels are correct

    accuracy = 100. * correct / train_data_size

    print('    [~ {}] Training set: Accuracy: {}/{} ({:.0f}%) - Maximum is {} bad batch repetitions'.format(time_since(epoch_start), correct, train_data_size,
                                                                      accuracy, max_batch_repetitions))

    return train_loss_history, accuracy, total_loss  # loss per epoch


def test(model, test_loader, criterion, print_loss_batchwise=False, use_cuda=True):
    """
    Implements one epoch of model testing.

    :param model: The nn.module implementation that is to be trained
    :param test_loader: DataLoader for the test data.
    :param criterion: Criteria to determine loss.
    :param print_loss_batchwise: Print loss for every batch instead of only for epoch.
    :return: Batchwise history of loss in testing.
    """
    epoch_start = time.time()

    # Make sure train functionality like dropout is deactivated.
    model.eval()

    if print_loss_batchwise:
        print("evaluating trained model ...")

    correct = 0
    test_data_size = len(test_loader.dataset)

    test_loss_history = []

    for stories, queries, answers, sl, ql in test_loader:
        stories = create_variable(stories.type(torch.LongTensor), use_cuda)
        queries = create_variable(queries.type(torch.LongTensor), use_cuda)
        answers = create_variable(answers.type(torch.LongTensor), use_cuda)
        sl = create_variable(sl.type(torch.LongTensor), use_cuda)
        ql = create_variable(ql.type(torch.LongTensor), use_cuda)

        output = model(stories, queries, sl, ql)

        loss = criterion(output, answers.view(-1))

        # Calculating elementwise loss  per batch
        test_loss_history.append(loss.data[0])

        pred_answers = output.data.max(1)[1]
        correct += pred_answers.eq(
                answers.data.view_as(pred_answers)).cpu().sum()  # calculate how many labels are correct

    accuracy = 100. * correct / test_data_size

    print('    [~ {}] Test set: Accuracy: {}/{} ({:.0f}%)'.format(time_since(epoch_start), correct, test_data_size,
                                                                  accuracy))

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


def load_data(voc_paths, train_paths, test_paths):
    voc = bd.Vocabulary()
    train_instances = []
    test_instances = []

    for voc_path in voc_paths:
        voc.extend_with_file(voc_path)

    for train_path in train_paths:
        train_instances += bd.BAbIInstance.instances_from_file(train_path)

    for test_path in test_paths:
        test_instances += bd.BAbIInstance.instances_from_file(test_path)

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


def plot_data_in_window(train_loss, test_loss, train_acc, test_acc):
    plt.figure()
    plt.plot(train_loss, label='train-loss', color='b')
    plt.plot(test_loss, label='test-loss', color='r')
    plt.xlabel("Batch")
    plt.ylabel("Average Elementwise Loss per Batch")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_acc, label='train-accuracy', color='b')
    plt.plot(test_acc, label='test-accuracy', color='r')
    plt.xlabel("Epoch")
    plt.ylabel("Correct answers in %")
    plt.legend()
    plt.show()  # Train cycle


def concatenated_params(params):
    params_str = [str(x) for x in params]
    params_str = reduce(lambda x, y: x + '_' + y, params_str)

    return params_str


def save_results(tasks, train_loss, test_loss, params, train_accuracy, test_accuracy, params_file, model, voc,
                 param_dict, make_plots=True):
    param_str = concatenated_params(params)

    tasks_str = "".join(["_qa" + str(task) for task in tasks])

    date = str(time.strftime("%Y:%m:%d:%H:%M:%S"))
    fname = "results/" + date.replace(":", "_") + "_" + param_str + "_tasks" + tasks_str + "/"
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

    if make_plots:
        plt.switch_backend('agg')
        plt.figure()
        plt.plot(train_loss, label='train-loss', color='b')
        plt.plot(test_loss, label='test-loss', color='r')
        plt.xlabel("Batch")
        plt.ylabel("Average Elementwise Loss per Batch")
        plt.legend()
        plt.savefig(fname + "loss_history.png")
        plt.figure()
        plt.plot(train_accuracy, label='train-accuracy', color='b')
        plt.plot(test_accuracy, label='test-accuracy', color='r')
        plt.xlabel("Epoch")
        plt.ylabel("Correct answers in %")
        plt.legend()
        plt.savefig(fname + "acc_history.png")
        plt.close("all")
    with open(fname + "params.txt", "w") as text_file:
        text_file.write(params_file)

    torch.save(model.state_dict(), fname + "trained_model.pth")
    pickle.dump(voc.voc_dict, open(fname + "vocabulary.pkl", "wb"))
    pickle.dump(param_dict, open(fname + "params.pkl", "wb"))


def create_variable(tensor, use_cuda=True):
    # Do cuda() before wrapping with variable
    if use_cuda and torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


if __name__ == "__main__":
    main()
