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

# Some utility functions
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads each sequence to the same length (length of the longest sequence).
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    # Raises
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format

    If only_supporting is true,
    only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.

    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    #flatten = lambda data: reduce(lambda x, y: x + y, data)
    for story, q, answer in data:
        data1 = [(story, q, answer) for story, q, answer in data if not max_length or len(story) <= max_length]
        data2 = [(story[len(story)-max_length:len(story)], q, answer) for story, q, answer in data if not max_length or len(story) > max_length]
        #if not max_length or len(story) <= max_length:
        #    data.append((story, q, answer))
        #else:
        #    data.append((story[len(story)-max_length:len(story)], q, answer))
    return data1+data2



def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    ys = []
    facts_lengths = []
    for stories, query, answer in data:
        xf = []
        for story in stories:
            x = [word_idx[w] for w in story]
            xf.append(x)

        xfl = [len(l) for l in xf]
        facts_lengths.append(np.array(xfl))
        xf = pad_sequences(xf, maxlen=10, padding='post')

        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        #y = np.zeros(len(word_idx) + 1)
        #y[word_idx[answer]] = 1
        #no one-hot-encoding for answer anymore!!
        y = word_idx[answer]
        xs.append(xf)
        xqs.append(xq)
        ys.append(y)
    xsl = [len(l) for l in xs]  #contains length of stories
    xqsl = [len(l) for l in xqs] # contains length of queries

    return pad_sequences(xs, maxlen=story_maxlen, padding='post'), pad_sequences(xqs, maxlen=query_maxlen, padding='post'), np.array(ys), np.array(xsl), np.array(xqsl), pad_sequences(facts_lengths, maxlen=story_maxlen, padding='post') # info pad_sequence wurde in rnn.py reinkopiert


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

class QAModel(nn.Module):
    def __init__(self, input_size, embedding_size, story_hidden_size, query_hidden_size, output_size, n_layers=1, bidirectional=False):
        super(QAModel, self).__init__()

        self.voc_size = input_size
        self.embedding_size = embedding_size
        self.story_hidden_size = story_hidden_size
        self.query_hidden_size = query_hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1

        self.story_embedding = nn.Embedding(input_size, embedding_size) #Embedding bildet ab von Vokabular (Indize) auf n-dim Raum

        self.story_rnn = nn.GRU(embedding_size, story_hidden_size, n_layers,
                                bidirectional=bidirectional, batch_first=True, dropout=0.3)

        self.fact_rnn = nn.GRU(embedding_size, story_hidden_size, n_layers,
                                bidirectional=bidirectional, batch_first=True, dropout=0.3)

        self.query_embedding = nn.Embedding(input_size, embedding_size)
        self.query_rnn = nn.GRU(embedding_size, query_hidden_size, n_layers,
                                bidirectional=bidirectional, batch_first=True, dropout=0.3)

        # info: if we use the old-forward function fc-layer has input-length: "story_hidden_size+query_hidden_size"
        self.fc = nn.Linear(story_hidden_size, output_size)
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
        s_e = self.story_embedding(story)   # encodings have size: batch_size*length_of_sequence*EMBBEDDING_SIZE

        # packed Story-Embeddings into RNN
        packed_story = torch.nn.utils.rnn.pack_padded_sequence(s_e, story_lengths.data.cpu().numpy(), batch_first=True)  # pack story
        story_output, story_hidden = self.story_rnn(packed_story, story_hidden)
        # unpacking is not necessary, because we use hidden states of RNN

        q_e = self.query_embedding(query)
        query_output, query_hidden = self.query_rnn(q_e, query_hidden)

        merged = torch.cat([story_hidden[0], query_hidden[0]],1)
        merged = merged.view(batch_size, -1)
        fc_output = self.fc(merged)
        sm_output = self.softmax(fc_output)

        return sm_output


    # new forward-function with question-code
    # achieves 100% on Task 1!!
    # --> question-code is like an attention-mechanism!
    def forward_question_code(self, story, query, story_lengths, query_lengths):

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
        question_code = question_code.view(batch_size,1,self.query_hidden_size)
        question_code = question_code.repeat(1,story.size(1),1)

        # Embed story
        s_e = self.story_embedding(story)

        # Combine story-embeddings with question_code
        combined = s_e + question_code

        # put combined tensor into story_rnn --> attention-mechansism through question_code
        packed_story = torch.nn.utils.rnn.pack_padded_sequence(combined, story_lengths.data.cpu().numpy(), batch_first=True)  # pack story
        story_output, story_hidden = self.story_rnn(packed_story, story_hidden)
        # remember: because we use the hidden states of the RNN, we don't have to unpack the tensor!

        # Do softmax on the encoded story tensor!
        fc_output = self.fc(story_hidden[0])
        sm_output = self.softmax(fc_output)

        return sm_output

    def forward(self, story, query, story_lengths, query_lengths, fact_lengths):

        #story: 32x20x10
        #query: 32x5
        #story_lengths: 32
        #query_lengths: 32
        #fact_lengths: 32x20

        #a = story[31]

        # Calculate Batch-Size
        batch_size = story.size(0)
        story_size = story.size(1)

        # Make a hidden
        fact_hidden = self._init_hidden(batch_size*story_size, self.story_hidden_size)
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
        question_code = question_code.view(batch_size,1,self.query_hidden_size)
        question_code = question_code.repeat(1,story.size(1),1)

        # Embed story
        s_e = self.story_embedding(story.view(batch_size,story_size*10))
        s_e = s_e.view(batch_size,story_size,10,-1)

        s_e = s_e.view(batch_size*story_size,10,-1)

        #s_e.view(32*20,10,-1)

        #packing = torch.nn.utils.rnn.pack_padded_sequence(s_e, story_lengths.data.cpu().numpy(), batch_first=True)  # pack story

        story_output, fact_hidden = self.fact_rnn(s_e.view(batch_size*story_size,10,-1), fact_hidden)

        fact_encodings = fact_hidden.view(batch_size, story_size, -1)

        # Combine story-embeddings with question_code
        combined = fact_encodings + question_code

        # put combined tensor into story_rnn --> attention-mechansism through question_code
        packed_story = torch.nn.utils.rnn.pack_padded_sequence(combined, story_lengths.data.cpu().numpy(), batch_first=True)  # pack story
        story_output, story_hidden = self.story_rnn(packed_story, story_hidden)
        # remember: because we use the hidden states of the RNN, we don't have to unpack the tensor!

        # Do softmax on the encoded story tensor!
        fc_output = self.fc(story_hidden[0])
        sm_output = self.softmax(fc_output)

        return sm_output

    def _init_hidden(self, batch_size, hidden_size):
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, hidden_size)
        return Variable(hidden)

# Train cycle
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

        output = model(stories, queries, sl, ql, fl)

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

    #if PRINT_LOSS:
    print('Training set: Accuracy: {}/{} ({:.0f}%)'.format(
        correct, train_data_size, accuracy))

    return train_loss_history, accuracy, total_loss  # loss per epoch

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
        #ql, perm_idx = ql.sort(0, descending=True) # if we sort query also --> then they do not fit together!
        fl = fl[perm_idx]
        ql = ql[perm_idx]
        queries = queries[perm_idx]
        answers = answers[perm_idx]

        output = model(stories, queries, sl, ql, fl)

        loss = criterion(output, answers)
        test_loss_history.append(loss.data[0])

        pred_answers = output.data.max(1)[1]
        correct += pred_answers.eq(answers.data.view_as(pred_answers)).cpu().sum() # calculate how many labels are correct

    accuracy = 100. * correct / test_data_size

    #if PRINT_LOSS:
    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
        correct, test_data_size, accuracy))

    return test_loss_history, accuracy


## Load data

data_path = "data/"

#challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
#challenge = 'tasks_1-20_v1-2/en/qa3_three-supporting-facts_{}.txt'
#challenge = 'tasks_1-20_v1-2/en/qa6_yes-no-questions_{}.txt'

print(challenge)

train_data = get_stories(open(data_path + challenge.format('train'), 'r'), max_length=20)
test_data = get_stories(open(data_path + challenge.format('test'), 'r'), max_length=20)

## Preprocess data

vocab = set()
flatten = lambda data: reduce(lambda x, y: x + y, data)
for story, q, answer in train_data + test_data:
    vocab |= set(flatten(story) + q + [answer])
vocab = sorted(vocab)

# Reserve 0 for masking via pad_sequences
# Vocabluary Size
vocab_size = len(vocab) + 1
#Creates Dictionary
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

#Max Length of Story and Query
story_maxlen = max(map(len, (x for x, _, _ in train_data + test_data)))
query_maxlen = max(map(len, (x for _, x, _ in train_data + test_data)))


## Parameters
EMBED_HIDDEN_SIZE = 50
STORY_HIDDEN_SIZE = 50
QUERY_HIDDEN_SIZE = 50  # note: since we are adding the encoded query to the embedded stories,
#  QUERY_HIDDEN_SIZE should be equal to EMBED_HIDDEN_SIZE
N_LAYERS = 1
BATCH_SIZE = 32
EPOCHS = 100
VOC_SIZE = vocab_size
LEARNING_RATE = 0.001 #0.0001

print('\nSettings:\nEMBED_HIDDEN_SIZE: %d\nSTORY_HIDDEN_SIZE: %d\nQUERY_HIDDEN_SIZE: %d'
      '\nN_LAYERS: %d\nBATCH_SIZE: %d\nEPOCHS: %d\nVOC_SIZE: %d\nLEARNING_RATE: %f\n\n'
      %(EMBED_HIDDEN_SIZE,STORY_HIDDEN_SIZE,QUERY_HIDDEN_SIZE,N_LAYERS,BATCH_SIZE,EPOCHS,VOC_SIZE,LEARNING_RATE))

PLOT_LOSS = True
PRINT_LOSS = True

## Create Test & Train-Data
x, xq, y, xl, xql, facts_lengths= vectorize_stories(train_data, word_idx, story_maxlen, query_maxlen)  # x: story, xq: query, y: answer, xl: story_lengths, xql: query_lengths
tx, txq, ty, txl, txql, t_facts_lengths = vectorize_stories(test_data, word_idx, story_maxlen, query_maxlen) # same naming but for test_data

train_dataset = QADataset(x,xq,y,xl,xql,facts_lengths)
train_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE, shuffle=True)

test_dataset = QADataset(tx,txq,ty,txl,txql,t_facts_lengths)
test_loader = DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE, shuffle=True)


## Initialize Model and Optimizer

model = QAModel(VOC_SIZE, EMBED_HIDDEN_SIZE, STORY_HIDDEN_SIZE, QUERY_HIDDEN_SIZE, VOC_SIZE, N_LAYERS)
#criterion = nn.CrossEntropyLoss()
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
    train_acc_history.append(train_accuracy) # = train_acc_history + [train_accuracy]
    test_acc_history.append(test_accuracy) # = test_acc_history + test_accuracy

# Plot Loss
if PLOT_LOSS:
    plt.figure()
    plt.hold(True)
    plt.plot(train_loss_history,'b')
    plt.plot(test_loss_history,'r')
    plt.show()

    plt.figure()
    plt.hold(True)
    plt.plot(train_acc_history,'b')
    plt.plot(test_acc_history,'r')
    plt.show()