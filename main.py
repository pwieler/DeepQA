from __future__ import print_function

import copy
import os
import pickle
import sys
import time
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import preprocessing.bAbIData as bd
from model.QAModel import QAModel
from model.QAModelLSTM import QAModelLSTM
from utils.utils import create_var, time_since, cuda_model
import pandas as pd
import random

random.seed(157)  # Set seed for reproduction
NoneType = type(None)


def main(task_i, validation=True):
    # Some old PY 2.6 hacks to include the dirs
    sys.path.insert(0, 'model/')
    sys.path.insert(0, 'preprocessing/')
    sys.path.insert(0, 'utils/')
    # Can be either 1,2,3 or 6 respective to the evaluated task.
    BABI_TASK = task_i

    print('Training for task: %d' % BABI_TASK)

    base_path = "data/tasks_1-20_v1-2/shuffled"  # shuffled

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

    PREVIOUSLY_TRAINED_MODEL = None
    ONLY_EVALUATE = False

    ## GridSearch Parameters
    EPOCHS = [40]  # Mostly you only want one epoch param, unless you want equal models with different training times.
    EMBED_HIDDEN_SIZES = [50]
    STORY_HIDDEN_SIZE = [100]
    N_LAYERS = [1]
    BATCH_SIZE = [16]
    LEARNING_RATE = [0.001]  # 0.0001

    ## Output parameters
    # Makes the training halt between every param set until you close the plot windows. Plots are saved either way.
    PLOT_LOSS_INTERACTIVE = False
    PRINT_BATCHWISE_LOSS = False

    grid_search_params = GridSearchParamDict(EMBED_HIDDEN_SIZES, STORY_HIDDEN_SIZE, N_LAYERS, BATCH_SIZE, LEARNING_RATE,
                                             EPOCHS)

    voc, train_instances, test_instances, validation_instances = load_data(babi_voc_path[BABI_TASK],
                                                                           babi_train_path[BABI_TASK],
                                                                           babi_test_path[BABI_TASK], validation)

    # Converts the words of the instances from string representation to integer representation using the vocabulary.
    vectorize_data(voc, train_instances, test_instances,
                   validation_instances)  # if validation Instances == none, nothing happens

    list_dicts = []
    list_top_results = []
    list_epoch_n = []
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

        train_loader, test_loader, validation_loader, train_final_loader = prepare_dataloaders(train_instances,
                                                                                               test_instances,
                                                                                               batch_size,
                                                                                               validation_instances=validation_instances)

        ## Initialize Model and Optimizer
        model = QAModel(voc_len, embedding_size, story_hidden_size, voc_len, n_layers)
        model = cuda_model(model)

        model_final = QAModel(voc_len, embedding_size, story_hidden_size, voc_len, n_layers)
        model_final = cuda_model(model_final)

        # If a path to a state dict of a previously trained model is given, the state will be loaded here.
        if PREVIOUSLY_TRAINED_MODEL is not None:
            model.load_state_dict(torch.load(PREVIOUSLY_TRAINED_MODEL))

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer_final = torch.optim.Adam(model_final.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()

        ptr_loader = test_loader  # Pointer for the validation/test loader
        if validation == True:
            ptr_loader = validation_loader

        train_loss, val_loss, train_acc, val_acc, eval_lists, test_acc_history, test_loss_history = conduct_training(model,model_final, train_final_loader, train_loader,
                                                                                ptr_loader, test_loader,
                                                                                optimizer,
                                                                                optimizer_final,
                                                                                criterion, only_evaluate=ONLY_EVALUATE,
                                                                                print_loss=PRINT_BATCHWISE_LOSS,
                                                                                epochs=epochs)

        evaluated_out = evaluate_outputs(eval_lists, voc)  # Evaluated output. Contains The Real Data

        params = [embedding_size, story_hidden_size, n_layers, batch_size, epochs, voc_len, learning_rate, epochs]

        save_results(BABI_TASK, train_loss, val_loss, params, train_acc, val_acc, readable_params, model, voc,
                     evaluated_out, test_loss_history, test_acc_history)

        # Plot Loss
        if PLOT_LOSS_INTERACTIVE:
            plot_data_in_window(train_loss, val_loss, train_acc, val_acc)


def replace_to_word(ids_vector, voc):
    st_list = []
    for i in range(len(ids_vector)):
        vec = [ids_vector[i]]
        vec = [voc.id_to_word(item) for item in vec]
        st = ' '.join(vec)
        st_list.append(st)
    return st_list

def replace_to_text_vec(ids_vector, voc):
    st_list = []
    for i in range(len(ids_vector)):
        vec = ids_vector[i, :]
        vec = vec.tolist()
        vec = [voc.id_to_word(item) for item in vec]
        st = ' '.join(vec)
        st_list.append(st)
    return st_list


def evaluate_outputs(eval_lists, voc):
    # Merge Batches
    stories = np.vstack([x[1] for x in eval_lists])
    GT = np.hstack([x[2] for x in eval_lists])
    story_l = np.hstack([x[3] for x in eval_lists])
    query_l = np.hstack([x[4] for x in eval_lists])
    answer = np.hstack([x[5] for x in eval_lists])
    queries = np.vstack([x[6] for x in eval_lists])
    best_3 = np.vstack([x[7] for x in eval_lists])
    best_5 = np.vstack([x[8] for x in eval_lists])
    tf = np.equal(GT, answer)

    answer_dist = np.vstack([tf, story_l, query_l])
    answer_dist = np.transpose(answer_dist)
    answer_dist = pd.DataFrame(answer_dist)
    answer_dist.columns = ["Correct", "Story Length", "Query Length"]
    stories_origin = replace_to_text_vec(stories, voc)
    stories_origin = pd.DataFrame(stories_origin)
    queries_origin = replace_to_text_vec(queries, voc)
    queries_origin = pd.DataFrame(queries_origin)
    best_3_origin = replace_to_text_vec(best_3, voc)
    best_3_origin = pd.DataFrame(best_3_origin)
    best_5_origin = pd.DataFrame(replace_to_text_vec(best_5, voc))
    answer = pd.DataFrame(replace_to_word(answer,voc))
    results = [["Answers Dis", "Original Stories", "Original Queries", "Best 3" , "Best 5", "Answers"], answer_dist, stories_origin, queries_origin, best_3_origin, best_5_origin, answer]
    return results


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

        stories = create_var(stories.type(torch.LongTensor))
        queries = create_var(queries.type(torch.LongTensor))
        answers = create_var(answers.type(torch.LongTensor))
        sl = create_var(sl.type(torch.LongTensor))
        ql = create_var(ql.type(torch.LongTensor))

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

        # Calculating elementwise loss per batch
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
        # TODO
        correct += pred_answers.eq(
            answers.data.view_as(pred_answers)).cpu().sum()  # calculate how many labels are correct

    accuracy = 100. * correct / train_data_size

    print('Training set: Accuracy: {}/{} ({:.0f}%)'.format(correct, train_data_size, accuracy))

    return train_loss_history, accuracy, total_loss  # loss per epoch

#This method is replaced by test-final for the ecaluation setting
def test(model, test_loader, criterion, PRINT_LOSS=False):
    model.eval()

    if PRINT_LOSS:
        print("evaluating trained model ...")

    correct = 0
    test_data_size = len(test_loader.dataset)

    test_loss_history = []
    stats_list = []
    for stories, queries, answers, sl, ql in test_loader:
        stories_np = stories.numpy()
        answers_np = answers.numpy()
        storyl_np = sl.numpy()
        queryl_np = ql.numpy()
        queries_np = queries.numpy()
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

        # Calculating elementwise loss  per batch
        test_loss_history.append(loss.data[0])

        pred_answers = output.data.topk(5)[1]
        predicted_answers_np = pred_answers.numpy()
        stats = [["stories", "Ground Truth", "story length", "Q lenght", "Predicted Answer", "Queries"], stories_np,
                 answers_np, storyl_np, queryl_np, predicted_answers_np, queries_np]
        stats_list.append(stats)
        correct += pred_answers.eq(
            answers.data.view_as(pred_answers)).cpu().sum()  # calculate how many labels are correct

    accuracy = 100. * correct / test_data_size

    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, test_data_size, accuracy))

    return test_loss_history, accuracy, stats_list


def test_final(model, test_loader, criterion, PRINT_LOSS=False):
    model.eval()

    if PRINT_LOSS:
        print("evaluating trained model ...")

    correct = 0
    test_data_size = len(test_loader.dataset)
    correct_best_3 = 0
    correct_best_5 = 0
    test_loss_history = []
    stats_list = []
    for stories, queries, answers, sl, ql in test_loader:
        stories_np = stories.numpy()
        answers_np = answers.numpy()
        storyl_np = sl.numpy()
        queryl_np = ql.numpy()
        queries_np = queries.numpy()
        stories = Variable(stories.type(torch.LongTensor))
        queries = Variable(queries.type(torch.LongTensor))
        answers = Variable(answers.type(torch.LongTensor))
        sl = Variable(sl.type(torch.LongTensor))
        ql = Variable(ql.type(torch.LongTensor))

        # Sort stories by their length
        sl, perm_idx = sl.sort(0, descending=True)
        stories = stories[perm_idx]
        ql = ql[perm_idx]
        queries = queries[perm_idx]
        answers = answers[perm_idx]
        answers_np = answers.data.numpy()
        stories_np = stories.data.numpy()
        output = model(stories, queries, sl, ql)

        loss = criterion(output, answers.view(-1))

        # Calculating elementwise loss  per batch
        test_loss_history.append(loss.data[0])

        pred_answers = output.data.max(1)[1]
        pred_top3 = output.data.topk(dim=1, k=3)[1]
        pred_top5 = output.data.topk(dim=1, k=5)[1]

        predicted_answers_np = pred_answers.numpy()

        pred_top3_np = pred_top3.numpy()
        pred_top5_np = pred_top5.numpy()

        top_3_status = [int(answers_np[i] in pred_top3_np[i, :].tolist()) for i in
                        range(len(answers_np))]  # Number of correct predictions in the top 3 predictions
        top_5_status = [int(answers_np[i] in pred_top5_np[i, :].tolist()) for i in range(len(answers_np))]

        stats = [["stories", "Ground Truth", "story length", "Q lenght", "Predicted Answer", "Queries", "Top 3 Answers",
                  "Top 5 answers"],
                 stories_np, answers_np, storyl_np, queryl_np, predicted_answers_np, queries_np, pred_top3_np,
                 pred_top5_np]
        stats_list.append(stats)
        correct += pred_answers.eq(
            answers.data.view_as(pred_answers)).cpu().sum()  # calculate how many labels are correct
        correct_best_3 += sum(top_3_status)
        correct_best_5 += sum(top_5_status)

    accuracy = 100. * correct / test_data_size
    accuracy_top3 = 100. * correct_best_3 / test_data_size
    accuracy_top5 = 100. * correct_best_5 / test_data_size

    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, test_data_size, accuracy))
    print('Test set: Accuracy (Best 3): {}/{} ({:.0f}%)'.format(correct_best_3, test_data_size, accuracy_top3))
    print('Test set: Accuracy (Best 5): {}/{} ({:.0f}%)'.format(correct_best_5, test_data_size, accuracy_top5))

    return test_loss_history, accuracy, stats_list, accuracy_top3, accuracy_top5


def prepare_dataloaders(train_instances, test_instances, batch_size, shuffle=True, validation_instances=None):
    tstr_len = max([len(inst.flat_story()) for inst in train_instances])
    tq_len = max([len(inst.question) for inst in train_instances])
    print(tq_len)
    if type(validation_instances) != NoneType:
        maxstr_val = max([len(inst.flat_story()) for inst in validation_instances])
        tstr_len = max(maxstr_val, tstr_len)

        maxq_val = max([len(inst.question) for inst in validation_instances])
        tq_len = max(maxq_val, tq_len)
        print(maxq_val)

        validation_dataset = bd.BAbiDataset(validation_instances, maxlen_str=tstr_len, maxlen_q=tq_len)

    train_dataset = bd.BAbiDataset(train_instances, maxlen_str=tstr_len, maxlen_q=tq_len)
    test_dataset = bd.BAbiDataset(test_instances, maxlen_str=tstr_len, maxlen_q=tq_len)

    train_final_instances = validation_instances.copy() + validation_instances.copy()
    train_final_dataset = bd.BAbiDataset(train_final_instances, maxlen_str=tstr_len, maxlen_q=tq_len)
    train_final_loader = DataLoader(dataset=train_final_dataset, batch_size=batch_size, shuffle=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = None
    if type(validation_instances) != NoneType:
        validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, validation_loader, train_final_loader


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
                                    "embedding_size": em,
                                    "story_hidden_size": s,
                                    "layers": l,
                                    "batch_size": b,
                                    "learning_rate": lr,
                                    "epochs": ep
                                })

        return self.params


def load_data(voc_path, train_path, test_path, split_validation=False):
    voc = bd.Vocabulary()
    train_instances = []
    test_instances = []

    voc.extend_with_file(voc_path)
    train_instances = bd.BAbIInstance.instances_from_file(train_path)
    test_instances = bd.BAbIInstance.instances_from_file(test_path)

    voc.sort_ids()

    validation_instances = None
    if split_validation:
        ids_l = list(range(len(train_instances)))
        val_size = int((len(ids_l) / 100) * 15)  # select 10% for validation
        val_sample = random.sample(ids_l, val_size)
        validation_instances = [train_instances[x] for x in val_sample]
        train_instances = [train_instances[i] for i in range(len(train_instances)) if i not in val_sample]

    return voc, train_instances, test_instances, validation_instances


def vectorize_data(voc, train_instances, test_instances, validation_instances=None):
    # At this point, training instances have been loaded with real word sentences.
    # Using the vocabulary we convert the words into integer representations, so they can converted with an embedding
    # later on.
    for inst1 in train_instances:
        inst1.vectorize(voc)



    for inst2 in test_instances:
        inst2.vectorize(voc)

    if type(validation_instances) != NoneType:
        for inst3 in validation_instances:
            inst3.vectorize(voc)


def conduct_training(model,model_final, train_final_loader, train_loader, validation_loader,test_loader, optimizer,optimizer_final, criterion,
                     only_evaluate=False,
                     print_loss=False,
                     epochs=1):
    train_loss_history = []
    validation_loss_history = []
    test_loss_history = []

    train_acc_history = []
    validation_acc_history = []
    test_acc_history = []

    eval_list = []

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
        validation_loss, validation_accuracy, eval_list, validation_accuracy_top_3, validation_accuracy_top_5 = test_final(
            model,
            validation_loader,
            criterion,
            PRINT_LOSS=False)

        # Add Loss to history
        validation_loss_history = validation_loss_history + validation_loss
        # Add Loss to history
        validation_acc_history.append(validation_accuracy)
    epochs_final = np.argmax(np.array(validation_acc_history))
    model.eval()
    for epoch in range(1, epochs_final + 1):
        print("Epoche: %d" % epoch)
        # Train cycle
        print("Test Final")
        train_loss, train_accuracy, total_loss = train(model_final, train_final_loader, optimizer_final, criterion, start,
                                                           epoch, False)

        # Test cycle
        test_loss, test_accuracy, eval_list_final, test_accuracy_top_3, test_accuracy_top_5 = test_final(model_final,
                                                                                                   test_loader,
                                                                                                   criterion,
                                                                                                   PRINT_LOSS=True)

        # Add Loss to history
        train_loss_history = train_loss_history + train_loss
        test_loss_history = test_loss_history + test_loss
        # Add Loss to history
        train_acc_history.append(train_accuracy)
        test_acc_history.append(test_accuracy)

    return train_loss_history, validation_loss_history, train_acc_history, validation_acc_history, eval_list_final, test_acc_history, test_loss_history


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


def save_results(task, train_loss, validation_loss, params, train_accuracy, validation_accuracy, params_file, model, voc,
                 eval_results, test_loss, test_accuracy, plots=True):
    param_str = concatenated_params(params)

    date = str(time.strftime("%Y:%m:%d:%H:%M:%S"))
    fname = "results/" + date.replace(":", "_") + "_" + param_str + "_task_" + str(task) + "/"
    try:
        os.stat(fname)
    except:
        os.mkdir(fname)
    tr_loss = np.array(train_loss)
    val_loss = np.array(validation_loss)
    tr_acc = np.array(train_accuracy)
    validation_acc = np.array(validation_accuracy)
    te_acc = np.array(test_accuracy)
    te_loss = np.array(test_loss)
    te_acc.tofile(fname + "test_accuracy.csv", sep=";")
    te_loss.tofile(fname + "test_loss.csv", sep=";")
    tr_loss.tofile(fname + "train_loss.csv", sep=";")
    val_loss.tofile(fname + "val_loss.csv", sep=";")
    tr_acc.tofile(fname + "train_accuracy.csv", sep=";")
    validation_acc.tofile(fname + "val_accuracy.csv", sep=";")
    eval_results[1].to_csv(fname + "distribution_answers.csv", sep=";")
    eval_results[2].to_csv(fname + "Stories.csv", sep=";")
    eval_results[3].to_csv(fname + "Queries.csv", sep=";")
    eval_results[4].to_csv(fname + "Best_3.csv", sep=";")
    eval_results[5].to_csv(fname + "Best_5.csv", sep=";")
    eval_results[6].to_csv(fname + "Answer.csv", sep=";")

    if plots == True:
        plt.figure()
        plt.plot(train_loss, label='train-loss', color='b')
        plt.plot(validation_loss, label='val-loss', color='r')
        plt.xlabel("Batch")
        plt.ylabel("Average Elementwise Loss per Batch")
        plt.legend()
        plt.savefig(fname + "loss_history.png")
        plt.figure()
        plt.plot(train_accuracy, label='train-accuracy', color='b')
        plt.plot(validation_accuracy, label='val-accuracy', color='r')
        plt.xlabel("Epoch")
        plt.ylabel("Correct answers in %")
        plt.legend()
        plt.savefig(fname + "acc_history.png")
        plt.close("all")
    with open(fname + "params.txt", "w") as text_file:
        text_file.write(params_file)

    torch.save(model.state_dict(), fname + "trained_model.pth")
    pickle.dump(voc.voc_dict, open(fname + "vocabulary.pkl", "wb"))


if __name__ == "__main__":
    for i in (1, 2, 3, 6):
        main(i)
