from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable


class SentenceModel(nn.Module):
    def __init__(self, input_size, embedding_size, story_hidden_size, query_hidden_size, output_size, n_layers=1, bidirectional=False):
        super(SentenceModel, self).__init__()

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

    def forward(self, story, query, story_lengths, query_lengths, fact_lengths, fact_maxlen):

        #story: 32x20x7
        #query: 32x5
        #story_lengths: 32
        #query_lengths: 32
        #fact_lengths: 32x20xfact_maxlen

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
        question_code_words = question_code.view(batch_size,1,1,self.query_hidden_size).repeat(1, story.size(1), fact_maxlen, 1)
        question_code_facts = question_code.repeat(1,story.size(1),1)

        # Embed story
        s_e = self.story_embedding(story.view(batch_size,story_size*fact_maxlen))
        s_e = s_e.view(batch_size,story_size,fact_maxlen,-1) # 32x20x7x50

        s_e = s_e + question_code_words

        fact_output, fact_hidden = self.fact_rnn(s_e.view(batch_size*story_size,fact_maxlen,-1), fact_hidden)

        fact_encodings = fact_hidden.view(batch_size, story_size, -1) # 32x20x50

        # Combine story-embeddings with question_code
        combined = fact_encodings + question_code_facts

        # put combined tensor into story_rnn --> attention-mechanism through question_code
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