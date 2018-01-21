from __future__ import print_function
from functools import reduce
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from preprocessing.preprocessing import time_since, generate_data, vectorize_data
from model.SentenceModel import SentenceModel

## Dataset for QA-Data
class QADataset(Dataset):

    def __init__(self, story, query, answer, story_lengths, query_lengths, facts_lengths):
        self.story = story
        self.query = query
        self.answer = answer
        self.story_lengths = story_lengths
        self.query_lengths = query_lengths
        self.facts_lengths = facts_lengths
        self.len = len(story)

    def __getitem__(self, index):
        return self.story[index], self.query[index], self.answer[index], self.story_lengths[index], self.query_lengths[index], self.facts_lengths[index]

    def __len__(self):
        return self.len

# Train-Cycle
def train():
    total_loss = 0
    correct = 0

    train_data_size = len(train_loader.dataset)

    train_loss_history = []

    model.train()

    for i, (stories, queries, answers, sl, ql, fl) in enumerate(train_loader, 1):

        stories = Variable(stories.type(torch.LongTensor))
        queries = Variable(queries.type(torch.LongTensor))
        answers = Variable(answers.type(torch.LongTensor))
        sl = Variable(sl.type(torch.LongTensor))
        ql = Variable(ql.type(torch.LongTensor))
        fl = Variable(fl.type(torch.LongTensor))

        # Sort stories by their length (because of packing in the forward step!)
        sl, perm_idx = sl.sort(0, descending=True)
        stories = stories[perm_idx]
        ql = ql[perm_idx]
        fl = fl[perm_idx]
        queries = queries[perm_idx]
        answers = answers[perm_idx]

        output = model(stories, queries, sl, ql, fl, fact_maxlen)

        loss = criterion(output, answers)

        total_loss += loss.data[0]

        train_loss_history.append(loss.data[0])

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if PRINT_LOSS:
            if i % 1 == 0:
                print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(
                    time_since(start), epoch, i *
                                              len(stories), len(train_loader.dataset),
                                              100. * i * len(stories) / len(train_loader.dataset),
                    loss.data[0]))

        pred_answers = output.data.max(1)[1]
        correct += pred_answers.eq(
            answers.data.view_as(pred_answers)).cpu().sum()  # calculate how many labels are correct

    accuracy = 100. * correct / train_data_size

    print('Training set: Accuracy: {}/{} ({:.0f}%)'.format(
        correct, train_data_size, accuracy))

    return train_loss_history, accuracy, total_loss  # loss per epoch

# Test-Cycle
def test():

    model.eval()

    if PRINT_LOSS:
        print("evaluating trained model ...")

    correct = 0
    test_data_size = len(test_loader.dataset)

    test_loss_history = []

    for stories, queries, answers, sl, ql, fl in test_loader:
        stories = Variable(stories.type(torch.LongTensor))
        queries = Variable(queries.type(torch.LongTensor))
        answers = Variable(answers.type(torch.LongTensor))
        sl = Variable(sl.type(torch.LongTensor))
        ql = Variable(ql.type(torch.LongTensor))
        fl = Variable(fl.type(torch.LongTensor))

        # Sort stories by their length
        sl, perm_idx = sl.sort(0, descending=True)
        stories = stories[perm_idx]
        fl = fl[perm_idx]
        ql = ql[perm_idx]
        queries = queries[perm_idx]
        answers = answers[perm_idx]

        output = model(stories, queries, sl, ql, fl, fact_maxlen)

        loss = criterion(output, answers)
        test_loss_history.append(loss.data[0])

        pred_answers = output.data.max(1)[1]
        correct += pred_answers.eq(answers.data.view_as(pred_answers)).cpu().sum() # calculate how many labels are correct

    accuracy = 100. * correct / test_data_size

    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
        correct, test_data_size, accuracy))

    return test_loss_history, accuracy


if __name__ == "__main__":

    ## Load data
    data_path = "data/"
    challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'

    # Max-Length = 20 means that we only consider the last 20 facts!
    train_data = generate_data(open(data_path + challenge.format('train'), 'r'), max_length=20)
    test_data = generate_data(open(data_path + challenge.format('test'), 'r'), max_length=20)


    ## Generate Vocabulary
    vocabulary = set()
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    for story, q, answer in train_data + test_data:
        vocabulary |= set(flatten(story) + q + [answer])
    vocabulary = sorted(vocabulary)

    # Vocabluary Size
    voc_size = len(vocabulary) + 1


    ## Generate Dictionary
    dictionary = dict((c, i + 1) for i, c in enumerate(vocabulary))

    # Maximum length of story, query and facts:
    story_maxlen = max(map(len, (x for x, _, _ in train_data + test_data)))
    query_maxlen = max(map(len, (x for _, x, _ in train_data + test_data)))
    fact_maxlen = 7


    ## Parameters
    EMBED_HIDDEN_SIZE = 50
    STORY_HIDDEN_SIZE = 50
    QUERY_HIDDEN_SIZE = 50
    # note: since we are adding the encoded query to the embedded stories,
    #  QUERY_HIDDEN_SIZE should be equal to EMBED_HIDDEN_SIZE

    N_LAYERS = 1
    BATCH_SIZE = 32
    EPOCHS = 50
    VOC_SIZE = voc_size
    LEARNING_RATE = 0.001

    print('\nSettings:\nEMBED_HIDDEN_SIZE: %d\nSTORY_HIDDEN_SIZE: %d\nQUERY_HIDDEN_SIZE: %d'
          '\nN_LAYERS: %d\nBATCH_SIZE: %d\nEPOCHS: %d\nVOC_SIZE: %d\nLEARNING_RATE: %f\n\n'
          %(EMBED_HIDDEN_SIZE,STORY_HIDDEN_SIZE,QUERY_HIDDEN_SIZE,N_LAYERS,BATCH_SIZE,EPOCHS,VOC_SIZE,LEARNING_RATE))

    PLOT_LOSS = True
    PRINT_LOSS = True


    ## Create Test & Train-Data
    train_stories, train_queries, train_answers, train_story_lengths, train_query_lengths, train_facts_lengths = \
        vectorize_data(train_data, dictionary, story_maxlen, query_maxlen, fact_maxlen)

    test_stories, test_queries, test_answers, test_story_lengths, test_query_lengths, test_facts_lengths = \
        vectorize_data(test_data, dictionary, story_maxlen, query_maxlen, fact_maxlen) # same naming but for test_data

    train_dataset = QADataset(train_stories, train_queries, train_answers, train_story_lengths, train_query_lengths, train_facts_lengths)
    train_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = QADataset(test_stories, test_queries, test_answers, test_story_lengths, test_query_lengths, test_facts_lengths)
    test_loader = DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE, shuffle=True)


    ## Initialize Model and Optimizer
    model = SentenceModel(VOC_SIZE, EMBED_HIDDEN_SIZE, STORY_HIDDEN_SIZE, QUERY_HIDDEN_SIZE, VOC_SIZE, N_LAYERS)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(model)


    ## Start training
    start = time.time()
    if PRINT_LOSS:
        print("Training for %d epochs..." % EPOCHS)

    train_loss_history = []
    test_loss_history = []

    train_acc_history = []
    test_acc_history = []

    for epoch in range(1, EPOCHS + 1):

        print("Epoche: %d" %epoch)
        # Train cycle
        train_loss, train_accuracy, total_loss = train()

        # Test cycle
        test_loss, test_accuracy = test()

        # Add Loss to history
        train_loss_history = train_loss_history+train_loss
        test_loss_history = test_loss_history+test_loss

        # Add Loss to history
        train_acc_history.append(train_accuracy)
        test_acc_history.append(test_accuracy)

    # Plot Loss
    if PLOT_LOSS:
        plt.figure()
        plt.plot(train_loss_history,'b')
        plt.plot(test_loss_history,'r')
        plt.show()

        plt.figure()
        plt.plot(train_acc_history,'b')
        plt.plot(test_acc_history,'r')
        plt.show()