from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.utils import create_var

## Our Basic Model
# This is our standard/basic model that reads in the story as a complete set of sentences.
# We use GRUs as RNNs for encoding the word-sequences!
# We have some Word-Embeddings to map form vocabulary to a linear subspace!
# Regularization is done with Dropout in the RNNs.
# More Dropout-Layers and more fully-connected Layers have not shown to be more efficient
# --> but there is still on-going work with parameter-tweaking!
class QAModel(nn.Module):
    def __init__(self, input_size, embedding_size, story_hidden_size, output_size, n_layers=1, bidirectional=False,
                 custom_embedding=None):
        super(QAModel, self).__init__()

        ## Definition of Input- & Output-Sizes
        self.voc_size = input_size
        self.embedding_size = embedding_size
        self.story_hidden_size = story_hidden_size
        self.query_hidden_size = embedding_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1

        ## Definition of Embeddings

        # Embedding is a mapping from Dimension of Vocabulary to Dimension of Embedding_Size
        # We are using separate story and query_embedding. We also tried to use both together,
        # but it did not perform better!
        self.story_embedding = custom_embedding if custom_embedding is not None else nn.Embedding(self.voc_size,
                                                                                                  self.embedding_size)

        self.query_embedding = custom_embedding if custom_embedding is not None else nn.Embedding(self.voc_size,
                                                                                                  self.embedding_size)

        ## Definition of RNNs (GRUs)
        self.story_rnn = nn.GRU(self.embedding_size, self.story_hidden_size, self.n_layers, bidirectional=bidirectional,
                                batch_first=True, dropout=0.3)

        self.query_rnn = nn.GRU(self.embedding_size, self.query_hidden_size, self.n_layers, bidirectional=bidirectional,
                                batch_first=True, dropout=0.3)

        ## Definition of Output-Layers --> here we do softmax on the vocabulary_size!

        # info: if we use the old-forward function fc-layer has input-length: "story_hidden_size+query_hidden_size"
        self.fc = nn.Linear(self.story_hidden_size, self.voc_size)
        self.softmax = nn.LogSoftmax()

    # This is the old forward version that we used before!
    # Below version with question_code performs much better!
    # difference:
    #       here we are not adding a question code before processing the story in the story_rnn
    #       this showed to be very efficient, because it adds a kind of attention mechanism
    def old_forward(self, story, query, story_lengths, query_lengths):

        #story: BATCH_SIZE x STORY_MAX_LEN
        #query: BATCH_SIZE x QUERY_MAX_LEN
        #story_lengths: contains the number of words each story would have without the padding
        #query_lengths: just needed to do packing/unpacking of queries --> this is not done at the moment

        # Determine batch-size
        batch_size = story.size(0)

        # Create hidden states for GRUs
        story_hidden = self._init_hidden(batch_size, self.story_hidden_size)
        query_hidden = self._init_hidden(batch_size, self.query_hidden_size)

        # Create Story-Embeddings
        # encodings have size: BATCH_SIZE*STORY_MAX_LEN*EMBBEDDING_SIZE
        s_e = self.story_embedding(story)

        # Pack the Story-Sequence. This is needed, because the sequences contain zero-padding!
        packed_story = torch.nn.utils.rnn.pack_padded_sequence(s_e, story_lengths.data.cpu().numpy(),
                                                               batch_first=True)  # pack story

        # Encode the story-sequence
        story_output, story_hidden = self.story_rnn(packed_story, story_hidden)
        # unpacking is not necessary, because we use hidden states of RNN

        # Create Query-Embeddings
        q_e = self.query_embedding(query)

        # Packing/Unpacking is not done with queries, because they have all the same lengths in our dataset.
        # Encode the query-sequence
        query_output, query_hidden = self.query_rnn(q_e, query_hidden)

        ## Concatenate the encodings of story and query to have both information combined!
        # Hint: This is the first time that story and query are combined
        #       --> so there was no possibility to optimize the encoding before according to the relevant query!
        #       --> see forward() for better solution with question_code!
        merged_encoding = torch.cat([story_hidden[0], query_hidden[0]], 1)
        merged_encoding = merged_encoding.view(batch_size, -1)

        ## Then do classification on that encoding!
        fc_output = self.fc(merged_encoding)
        sm_output = self.softmax(fc_output)

        return sm_output

    ## New forward-function with Question-Code
    # achieves 100% on Task 1, other Tasks also improved significantly!
    # --> question-code is like an attention-mechanism!
    def forward(self, story, query, story_lengths, query_lengths):

        # Determine Batch-Size
        batch_size = story.size(0)

        # Create hidden states for GRUs
        story_hidden = self._init_hidden(batch_size, self.story_hidden_size)
        query_hidden = self._init_hidden(batch_size, self.query_hidden_size)

        # Embed Query-Words
        q_e = self.query_embedding(query)

        # Generate an encoding for the Query-Sequence with RNN
        query_output, query_hidden = self.query_rnn(q_e, query_hidden)

        # question_code contains the encoded question!
        # --> we give this directly into the story_rnn,
        # so that the story_rnn can focus on the question already
        # and can forget unnecessary information!
        question_code = query_hidden[0]
        question_code = question_code.view(batch_size, 1, self.query_hidden_size)

        ## Question-Code has to have the same size as the story-embeddings!
        # --> then we can just add both!
        question_code = question_code.repeat(1, story.size(1), 1)

        # Embed Story-Words
        s_e = self.story_embedding(story)

        # Combine Story-Embeddings with Question-Code
        combined = s_e + question_code

        # Put combined tensor into story_rnn --> kind of attention-mechanism through question_code
        packed_story = torch.nn.utils.rnn.pack_padded_sequence(combined, story_lengths.data.cpu().numpy(),
                                                               batch_first=True)  # pack story
        story_output, story_hidden = self.story_rnn(packed_story, story_hidden)
        # remember: because we use the hidden states of the RNN, we don't have to unpack the tensor!

        # Do Softmax-Classification on the encoded Story-Tensor!
        fc_output = self.fc(story_hidden[0])
        sm_output = self.softmax(fc_output)

        return sm_output

    ## This method generates hidden states for the RNNs for first step!
    def _init_hidden(self, batch_size, hidden_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, hidden_size)
        return create_var(hidden)
