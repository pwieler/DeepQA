from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


class QAFFModel(nn.Module):
    """
        This model is just to visualize how bad the ff network performs on the task(simple benchmark). Do not use in practice. 
    """
    # This modell overfits greatly! Not suitable for the problem, but good to illustrate why we use the RNN!
    def __init__(self, input_size, embedding_size, story_hidden_size, query_hidden_size, output_size, n_layers=1,
                 bidirectional=False, s_len=-1, q_len=-1, custom_embedding=None):
        super(QAFFModel, self).__init__()
        assert (s_len > 1)
        assert (q_len > 1)
        self.voc_size = input_size
        self.embedding_size = embedding_size
        self.story_hidden_size = story_hidden_size
        self.query_hidden_size = query_hidden_size
        self.n_layers = n_layers
        self.s_len = s_len
        self.q_len = q_len
        self.story_embedding = custom_embedding if custom_embedding is not None else nn.Embedding(input_size,
                                                                                                  embedding_size)
        # Embedding bildet ab von Vokabular (Indize) auf n-dim Raum

        self.query_embedding = custom_embedding if custom_embedding is not None else nn.Embedding(input_size,
                                                                                                  embedding_size)

        # info: if we use the old-forward function fc-layer has input-length: "story_hidden_size+query_hidden_size"
        fc1o = int(np.floor(0.5 * embedding_size * (self.q_len + self.s_len)))
        self.fc1 = nn.Linear(embedding_size * (self.q_len + self.s_len), fc1o)
        init.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
        init.constant(self.fc1.bias, 0.1)
        self.fc1a = nn.Tanh()
        self.dropo = nn.Dropout()
        self.fc2 = nn.Linear(fc1o, output_size)
        init.xavier_uniform(self.fc2.weight, gain=np.sqrt(2))
        init.constant(self.fc2.bias, 0.1)
        self.softmax = nn.LogSoftmax()
        # this is the old forward version! below version with question_code performs much better!!

    def forward(self, story, query, story_lengths, query_lengths):
        # input shape: B x S (input size)

        # story has dimension batch_size * number of words
        batch_size = story.size(0)

        # Create Story-Embeddings
        s_e = self.story_embedding(story)  # encodings have size: batch_size*length_of_sequence*EMBBEDDING_SIZE

        # Create Question embedding
        q_e = self.query_embedding(query)

        # Transform the tensors to do the processing
        s_e = s_e.view(batch_size, -1)
        q_e = q_e.view(batch_size, -1)
        merged = torch.cat([s_e, q_e], 1)

        # First fc with tanh
        fc_output = self.fc1(merged)
        th_out = self.fc1a(fc_output)

        # Apply dropout
        th_out1 = self.dropo(th_out)
        out = self.fc2(th_out1)
        sm_output = self.softmax(out)

        return sm_output
