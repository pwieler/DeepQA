from __future__ import print_function
from functools import reduce
import re

import time
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import math

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
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data



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
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        #y = np.zeros(len(word_idx) + 1)
        #y[word_idx[answer]] = 1
        #no one-hot-encoding for answer anymore!!
        y = word_idx[answer]
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    xsl = [len(l) for l in xs]  #contains length of stories
    xqsl = [len(l) for l in xqs] # contains length of queries

    return pad_sequences(xs, maxlen=story_maxlen, padding='post'), pad_sequences(xqs, maxlen=query_maxlen, padding='post'), np.array(ys), np.array(xsl), np.array(xqsl) # info pad_sequence wurde in rnn.py reinkopiert


class QADataset(Dataset):

    def __init__(self, story, query, answer, story_lengths, query_lengths):
        self.story = story
        self.query = query
        self.answer = answer
        self.story_lengths = story_lengths
        self.query_lengths = query_lengths
        self.len = len(story)

    def __getitem__(self, index):
        return self.story[index], self.query[index], self.answer[index], self.story_lengths[index], self.query_lengths[index]

    def __len__(self):
        return self.len


def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


class QAModel(nn.Module):
    # Our model

    def __init__(self, input_size, embedding_size, story_hidden_size, query_hidden_size, output_size, n_layers=1, bidirectional=False):
        super(QAModel, self).__init__()

        self.embedding_size = embedding_size
        self.story_hidden_size = story_hidden_size
        self.query_hidden_size = query_hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1

        self.story_embedding = nn.Embedding(input_size, embedding_size) #Embedding bildet ab von Vokabular (Indize) auf n-dim Raum
        # self.dropout_1 = nn.Dropout(0.3)
        self.story_rnn = nn.GRU(embedding_size, story_hidden_size, n_layers,
                                bidirectional=bidirectional)

        self.query_embedding = nn.Embedding(input_size, embedding_size)
        # self.dropout_2 = nn.Dropout(0.3)
        self.query_rnn = nn.GRU(embedding_size, query_hidden_size, n_layers,
                                bidirectional=bidirectional)

        self.fc = nn.Linear(story_hidden_size+query_hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, story, query, story_lengths, query_lengths):
        # Note: we run this all at once (over the whole input sequence)
        # input shape: B x S (input size)
        # transpose to make S(sequence) x B (batch)
        story = story.t()
        query = query.t()

        # story hat jetzt Dimension Sequenzlänge x Batchgröße --> z.B. 552x32 Wörter
        batch_size = story.size(1)

        # Make a hidden
        story_hidden = self._init_hidden(batch_size, self.story_hidden_size)
        #print(story_hidden.size())
        query_hidden = self._init_hidden(batch_size, self.query_hidden_size)

        #print(story.size())
        s_e = self.story_embedding(story)   # jedes einzelne Wort wird in das Embedding abgebildet, deshalb hat man nun 552x32
                                            # Embeddings der Größe EMBBEDDING_SIZE. --> 552x32xEMBBEDDING_SIZE
        #print(s_e.size())
        # s_e = self.dropout_1(s_e)
        packed_story = torch.nn.utils.rnn.pack_padded_sequence(s_e, story_lengths.data.cpu().numpy())  # pack story
        story_output, story_hidden = self.story_rnn(packed_story, story_hidden)
        unpacked_story, unpacked_story_len = torch.nn.utils.rnn.pad_packed_sequence(story_output)  # unpack story

        #print(query.size())
        q_e = self.query_embedding(query)
        #print(q_e.size())
        # q_e = self.dropout_2(q_e)
        packed_query = torch.nn.utils.rnn.pack_padded_sequence(q_e, query_lengths.data.cpu().numpy())  # pack query
        query_output, query_hidden = self.query_rnn(packed_query, query_hidden)
        unpacked_query, unpacked_query_len = torch.nn.utils.rnn.pad_packed_sequence(query_output)  # unpack query

        #print('a')
        #print(unpacked_story[-1].size())
        #print(packed_story.size())
        #print(story_output.size())
        #print('b')
        #print(unpacked_query[-1].size())
        #print(packed_query.size())
        #print(query_output.size())

        merged = torch.cat([unpacked_story[-1], unpacked_query[-1]],1)
        fc_output = self.fc(merged)
        sm_output = self.softmax(fc_output)

        return sm_output

    def _init_hidden(self, batch_size, hidden_size):
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, hidden_size)
        return create_variable(hidden)


data_path = "data/"

challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'

train = challenge.format('train')
test = challenge.format('test')

path = data_path+train

train = get_stories(open(data_path+challenge.format('train'), 'r'))
test = get_stories(open(data_path+challenge.format('test'), 'r'))

vocab = set()
for story, q, answer in train + test:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)

# Reserve 0 for masking via pad_sequences
# Vocabluary Size
vocab_size = len(vocab) + 1
#Creates Dictionary
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

#Max Length of Story and Query
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
query_maxlen = max(map(len, (x for _, x, _ in train + test)))

# Parameters and DataLoaders
EMBED_HIDDEN_SIZE = 50
STORY_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 40
VOC_SIZE = vocab_size

x, xq, y, xl, xql,= vectorize_stories(train, word_idx, story_maxlen, query_maxlen)  # x: story, xq: query, y: answer, xl: story_lengths, xql: query_lengths
tx, txq, ty, txl, txql = vectorize_stories(test, word_idx, story_maxlen, query_maxlen) # same naming but for test_data

train_dataset = QADataset(x,xq,y,xl,xql)
train_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE, shuffle=True)

test_dataset = QADataset(tx,txq,ty,txl,txql)
test_loader = DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE, shuffle=True)

model = QAModel(VOC_SIZE, EMBED_HIDDEN_SIZE, STORY_HIDDEN_SIZE, QUERY_HIDDEN_SIZE, VOC_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# Train cycle
def train():
    total_loss = 0

    loss_history = []

    for i, (stories, queries, answers, sl, ql) in enumerate(train_loader, 1):

        stories = Variable(stories.type(torch.LongTensor))
        queries = Variable(queries.type(torch.LongTensor))
        answers = Variable(answers.type(torch.LongTensor))
        sl = Variable(sl.type(torch.LongTensor))
        ql = Variable(ql.type(torch.LongTensor))

        # Sort tensors by their length
        sl, perm_idx = sl.sort(0, descending=True)
        stories = stories[perm_idx]
        ql, perm_idx = ql.sort(0, descending=True)
        queries = queries[perm_idx]

        output = model(stories, queries, sl, ql)

        loss = criterion(output, answers)

        # print(input.size())
        # print(seq_lengths.size())
        # print(output.size())
        # print(target.size())

        total_loss += loss.data[0]

        loss_history.append(loss.data[0])

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1 == 0:
            print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(
                time_since(start), epoch, i *
                                          len(stories), len(train_loader.dataset),
                                          100. * i * len(stories) / len(train_loader.dataset),
                                          loss.data[0]))

    return loss_history, total_loss # loss per epoch


def test():

    print("evaluating trained model ...")
    correct = 0
    test_data_size = len(test_loader.dataset)

    for stories, queries, answers, sl, ql in test_loader:
        #for names, countries in test_loader:
        stories = Variable(stories.type(torch.LongTensor))
        queries = Variable(queries.type(torch.LongTensor))
        answers = Variable(answers.type(torch.LongTensor))
        sl = Variable(sl.type(torch.LongTensor))
        ql = Variable(ql.type(torch.LongTensor))

        # Sort tensors by their length
        sl, perm_idx = sl.sort(0, descending=True)
        stories = stories[perm_idx]
        ql, perm_idx = ql.sort(0, descending=True)
        queries = queries[perm_idx]

        output = model(stories, queries, sl, ql)

        pred = output.data.max(1)[1]
        correct += pred.eq(answers.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, test_data_size, 100. * correct / test_data_size))


start = time.time()
print("Training for %d epochs..." % EPOCHS)

l_history = []

for epoch in range(1, EPOCHS + 1):

    # Train cycle
    epoch_history, total_loss = train()
    test()

    l_history = l_history+epoch_history

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(l_history)
    plt.show()




