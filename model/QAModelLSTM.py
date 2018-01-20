from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd


class QAModelLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, story_hidden_size, output_size, n_layers=1, bidirectional=False,
                 custom_embedding=None):
        super(QAModelLSTM, self).__init__()

        self.voc_size = input_size
        self.embedding_size = embedding_size
        self.story_hidden_size = story_hidden_size
        self.query_hidden_size = embedding_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1

        # Embedding bildet ab von Vokabular (Indize) auf n-dim Raum
        self.story_embedding = custom_embedding if custom_embedding is not None else nn.Embedding(self.voc_size,
                                                                                                  self.embedding_size)

        self.story_rnn = nn.LSTM(self.embedding_size, self.story_hidden_size, self.n_layers, bidirectional=bidirectional,
                                batch_first=True, dropout=0.3)

        self.query_embedding = custom_embedding if custom_embedding is not None else nn.Embedding(self.voc_size,
                                                                                                  self.embedding_size)

        self.query_rnn = nn.LSTM(self.embedding_size, self.query_hidden_size, self.n_layers, bidirectional=bidirectional,
                                batch_first=True, dropout=0.3)

        # info: if we use the old-forward function fc-layer has input-length: "story_hidden_size+query_hidden_size"
        self.fc = nn.Linear(self.story_hidden_size, self.voc_size)
        self.softmax = nn.LogSoftmax()

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

        # put combined tensor into story_rnn --> attention-mechanism through question_code
        packed_story = torch.nn.utils.rnn.pack_padded_sequence(combined, story_lengths.data.cpu().numpy(),
                                                               batch_first=True)  # pack story
        story_output, story_hidden = self.story_rnn(packed_story, story_hidden)
        # remember: because we use the hidden states of the RNN, we don't have to unpack the tensor!

        # Do softmax on the encoded story tensor!
        fc_output = self.fc(story_hidden[-1].view(batch_size,self.story_hidden_size))
        sm_output = self.softmax(fc_output)

        return sm_output

    def _init_hidden(self, batch_size, hidden_size):
        return (autograd.Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, hidden_size)),
                autograd.Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, hidden_size)))
