from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

## Model with separated facts
#  Remember:
#       Our dataset consists out of stories and queries --> every story belongs to one query
#       --> every story consists out of multiple facts
#       --> one fact is one sentence: "Mary is in the house."
#  This is a model, where all the facts for one query are read in separateley instead of reading in all words together!
#  The intention was that it may perform better because it has more information about one unique fact!
#  At the moment it does not outperform our standard model --> but here it is working progress!
#
class SentenceModel(nn.Module):
    def __init__(self, input_size, embedding_size, story_hidden_size, query_hidden_size, output_size, n_layers=1, bidirectional=False):
        super(SentenceModel, self).__init__()

        ## Definition of Input- & Output-Sizes
        self.voc_size = input_size
        self.embedding_size = embedding_size
        self.story_hidden_size = story_hidden_size
        self.query_hidden_size = query_hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1

        ## Definition of Embeddings
        self.story_embedding = nn.Embedding(input_size, embedding_size)

        self.query_embedding = nn.Embedding(input_size, embedding_size)

        ## Definition of RNNs --> we have three different GRUs
        # Reads in each word of one fact --> generates a encoding for all the facts belonging to one query
        self.fact_rnn = nn.GRU(embedding_size, story_hidden_size, n_layers,
                                bidirectional=bidirectional, batch_first=True, dropout=0.4)

        # Reads in all fact-encodings belonging to one query --> generates one encoding for the whole story that is related to the query!
        self.story_rnn = nn.GRU(embedding_size, story_hidden_size, n_layers,
                                bidirectional=bidirectional, batch_first=True, dropout=0.4)

        # Reads in each word of the query --> generates one encoding for the query
        self.query_rnn = nn.GRU(embedding_size, query_hidden_size, n_layers,
                                bidirectional=bidirectional, batch_first=True, dropout=0.4)

        ## Definition of Output-Layers --> here we do softmax on the vocabulary_size!
        self.fc1 = nn.Linear(story_hidden_size, 300)
        self.dr1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(300, 250)
        self.dr2 = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(250, output_size)
        self.dr3 = nn.Dropout(p=0.4)
        self.softmax = nn.LogSoftmax()

    def forward(self, story, query, story_lengths, query_lengths, fact_lengths, fact_maxlen):
        #story: BATCH_SIZE x STORY_MAX_LEN x FACT_MAX_LEN
        #query: BATCH_SIZE x QUERY_MAX_LEN
        #story_lengths: contains the number of facts each story has
        #query_lengths: just needed to do packing/unpacking of queries --> this is not done at the moment
        #fact_lengths: just needed to do packing/unpacking of facts --> this is not done at the moment

        # FACT_MAX_LEN is the maximum size of one fact (maximum amount of words + padding)

        # Determine Batch-Size
        batch_size = story.size(0)
        story_size = story.size(1)

        # Make hidden for the GRUs
        fact_hidden = self._init_hidden(batch_size*story_size, self.story_hidden_size)
        story_hidden = self._init_hidden(batch_size, self.story_hidden_size)
        query_hidden = self._init_hidden(batch_size, self.query_hidden_size)

        # Embed Query-Words
        q_e = self.query_embedding(query)
        # Encode query-sequence with RNN
        query_output, query_hidden = self.query_rnn(q_e, query_hidden)

        # question_code contains the encoded question!
        # --> we give this directly into the story_rnn,
        # so that the story_rnn can focus on the question already
        # and can forget unnecessary information!
        question_code = query_hidden[0]
        question_code = question_code.view(batch_size,1,self.query_hidden_size)

        # Create a question-code for every word!
        question_code_words = question_code.view(batch_size,1,1,self.query_hidden_size).repeat(1, story.size(1), fact_maxlen, 1)
        # Create a question-code for every fact!
        question_code_facts = question_code.repeat(1,story.size(1),1)

        # Embed Words that are contained in the story
        # --> to do that we have to rearrange the tensor, so that we have the form:
        #       Batch_size x #Words
        s_e = self.story_embedding(story.view(batch_size,story_size*fact_maxlen))
        s_e = s_e.view(batch_size,story_size,fact_maxlen,-1) # 32x20x7x50

        # Combine word-embeddings with question_code
        s_e = s_e + question_code_words

        # Read in the words belonging to the facts into the Fact-RNN --> generate fact-encodings
        fact_output, fact_hidden = self.fact_rnn(s_e.view(batch_size*story_size,fact_maxlen,-1), fact_hidden)

        fact_encodings = fact_hidden.view(batch_size, story_size, -1) # 32x20x50

        # Combine story-embeddings with question_code
        combined = fact_encodings + question_code_facts

        # put combined tensor into story_rnn --> attention-mechanism through question_code
        packed_story = torch.nn.utils.rnn.pack_padded_sequence(combined, story_lengths.data.cpu().numpy(), batch_first=True)  # pack story
        story_output, story_hidden = self.story_rnn(packed_story, story_hidden)
        # remember: because we use the hidden states of the RNN, we don't have to unpack the tensor!

        # Do softmax on the encoded story tensor!
        fc1_out = F.relu(self.fc1(story_hidden[0]))
        fc1_out = self.dr1(fc1_out)
        fc2_out = F.relu(self.fc2(fc1_out))
        fc2_out = self.dr2(fc2_out)
        fc3_out = F.relu(self.fc3(fc2_out))
        fc3_out = self.dr3(fc3_out)

        sm_output = self.softmax(fc3_out)

        return sm_output

    def _init_hidden(self, batch_size, hidden_size):
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, hidden_size)
        return Variable(hidden)