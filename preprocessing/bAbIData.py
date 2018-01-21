from typing import List
import re
import torch
import torch.autograd as autograd
import torch.nn as nn
import math
import copy
from torch.utils.data import Dataset
import numpy as np


class Vocabulary:
    def __init__(self, file=None, vocabulary_dict=None, embedding=None):
        self.voc_dict = vocabulary_dict if vocabulary_dict is not None else dict()
        self.embedding = embedding

        self.voc_dict["<pad>"] = 0

        if file is not None:
            self.extend_with_file(file)

    def __len__(self):
        return len(self.voc_dict)

    def __repr__(self):
        rep = "Vocabulary:"

        for tuple in sorted(self.voc_dict.items(), key=lambda a: a[1]):
            rep += " " + str(tuple)

        return rep

    def word_to_id(self, word):
        # This is a trick entry for words that are non existent.
        if word is None or word not in self.voc_dict:
            return len(self.voc_dict)

        return self.voc_dict[word]

    def words_to_ids(self, words):
        return [self.word_to_id(word) for word in words]

    def word_to_tensor(self, word):
        if self.embedding is None:
            self.initialize_embedding()

        return self.embedding(autograd.Variable(torch.LongTensor([self.word_to_id(word)])))

    def extend_with_word(self, word):
        if word not in self.voc_dict:
            self.voc_dict[word] = len(self.voc_dict)

    def extend_with_text(self, text):
        word_set = set()

        for word_blob in text.split():
            alnum_word = "".join([c for c in word_blob if c.isalpha()])
            symbols = "".join([c for c in word_blob if not c.isalnum()])

            if len(alnum_word) > 0:
                if alnum_word not in word_set:
                    word_set.add(alnum_word)

            if len(symbols) > 0:
                if symbols not in word_set:
                    word_set.add(symbols)

        for word in word_set:
            self.extend_with_word(word)

    def sort_ids(self):
        i = 1

        for key in sorted(self.voc_dict.keys()):
            if self.voc_dict[key] is not 0:
                self.voc_dict[key] = i
                i += 1

    def extend_with_lines(self, lines):
        """
        Extends this vocabulary by using the text in the list of text lines given.

        :param lines: List of lines of text that will be used for extending the vocabulary
        """
        for line in lines:
            self.extend_with_text(line)

    def extend_with_file(self, path="data/qa2_two-supporting-facts_train.txt"):
        with open(path, 'r') as f:
            self.extend_with_lines(f.readlines())

    def initialize_embedding(self, em_dim=-1):
        if em_dim is -1:
            em_dim = int(math.log2(len(self.voc_dict) + 1))

        self.embedding = nn.Embedding(num_embeddings=len(self.voc_dict) + 1, embedding_dim=em_dim)


class BAbIInstance:
    def __init__(self):
        self.indexed_story = []
        self.question = []
        self.answer = []
        self.hints = []

    def __repr__(self):
        rep = "BAbIInstance:\n" + "  Story:\n"

        for line in self.indexed_story:
            rep += "    " + str(line[0]) + " " + str(line[1]) + "\n"

        rep += "  Question:\n" + "    " + str(self.question) + "\n"
        rep += "  Answer:\n" + "    " + str(self.answer) + "\n"

        rep += "  Hint sentences:\n"

        for line in self.hint_sentences():
            rep += "    " + str(line[0]) + " " + str(line[1]) + "\n"

        return rep

    def flat_story(self):
        flat_story = []

        for sentence in self.indexed_story:
            flat_story += sentence[1]

        return flat_story

    def question(self):
        return self.question

    def answer(self):
        return self.answer

    def hint_sentences(self):
        sentences = []

        for hint in sorted(self.hints):
            for sentence in self.indexed_story:
                if sentence[0] == hint:
                    sentences.append(sentence)

        return sentences

    def vectorize(self, voc):
        for s in self.indexed_story:
            s[1] = voc.words_to_ids(s[1])

        self.question = voc.words_to_ids(self.question)
        self.answer = voc.words_to_ids(self.answer)

    @staticmethod
    def instances_from_file(path):
        training_instances = []

        with open(path, 'r') as f:
            training_instances += BAbIInstance.instances_from_lines(f.readlines())

            return training_instances

    @staticmethod
    def instances_from_lines(lines):
        training_instances = []

        ind_lines = BAbIInstance._indexed_lines(lines)

        # Either there is no story or the story begins with a question
        if len(ind_lines) is 0 or len(ind_lines[0]) > 2:
            return []

        instance = BAbIInstance()

        current_story = []

        for line in ind_lines:
            if line[0] is 1:
                current_story = []

            if len(line) < 3:
                current_story.append([line[0], line[1]])
            else:
                instance.indexed_story = copy.deepcopy(current_story)
                instance.question = copy.deepcopy(line[1])
                instance.answer = [line[2]]
                instance.hints = copy.deepcopy(line[3])

                training_instances.append(instance)

                instance = BAbIInstance()

        return training_instances

    @staticmethod
    def _indexed_lines(lines):
        """
        Tokenize the stories and their information for easy access.

        :param lines: The lines of a QA babi file.

        :return: Returns a list of lists which each contain the story line number,
                    a list of the tokenized words in the sentence and in those lines where it applies,
                    the answer to the question and the hint line numbers.

                    Example: [[9, ['Daniel', 'went', 'back', 'to', 'the', 'kitchen', '.']], [10, ['Where', 'is',
                    'the', 'football', '?'], 'kitchen', [6, 9]]]
        """
        # First trim index
        indexed_lines = []
        for line in lines:
            splitter = line.split(maxsplit=1)
            indexed_lines.append([int(splitter[0]), splitter[1]])

        for line in indexed_lines:
            if '?' in line[1]:
                splitter = re.split('(\?)', line[1])
                sentence = splitter[0] + splitter[1]  # The sentence including questionmark

                answer = splitter[2].split("\t")[1]
                supporting_facts = [int(st) for st in splitter[2].split("\t")[2].split()]

                line[1] = sentence
                line.append(answer)
                line.append(supporting_facts)

        for line in indexed_lines:
            line[1] = [st for st in re.split("(\W)", line[1]) if st not in ["", " ", "\n"]]

        return indexed_lines


class BAbiDataset(Dataset):
    def __init__(self, instances, pad_sequences=True):
        self.instances = instances
        self.pad_sequences = pad_sequences
        self.maxlen_story = max([len(inst.flat_story()) for inst in instances])
        self.maxlen_question = max([len(inst.question) for inst in instances])

    def __getitem__(self, index):
        out_answer = self.instances[index].answer[0]
        out_story_len = len(self.instances[index].flat_story())
        out_question_len = len(self.instances[index].question)

        if self.pad_sequences:
            out_story = np.pad(self.instances[index].flat_story(), pad_width=(0, self.maxlen_story - out_story_len),
                               mode='constant', constant_values=0)
            out_question = np.pad(self.instances[index].question, pad_width=(0, self.maxlen_question - out_question_len),
                               mode='constant', constant_values=0)
        else:
            out_story = np.array(self.instances[index].flat_story())
            out_question = np.array(self.instances[index].question)

        return out_story, out_question, out_answer, out_story_len, out_question_len

    def __len__(self):
        return len(self.instances)


def main():
    voc = Vocabulary()
    voc.extend_with_file("data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt")
    voc.extend_with_file("data/tasks_1-20_v1-2/en/qa2_two-supporting-facts_train.txt")
    voc.extend_with_file("data/tasks_1-20_v1-2/en/qa3_three-supporting-facts_train.txt")
    voc.extend_with_file("data/tasks_1-20_v1-2/en/qa6_yes-no-questions_train.txt")

    training_instances = []
    training_instances += BAbIInstance.instances_from_file(
            "data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt")
    training_instances += BAbIInstance.instances_from_file("data/tasks_1-20_v1-2/en/qa2_two-supporting-facts_train.txt")
    training_instances += BAbIInstance.instances_from_file(
            "data/tasks_1-20_v1-2/en/qa3_three-supporting-facts_train.txt")
    training_instances += BAbIInstance.instances_from_file("data/tasks_1-20_v1-2/en/qa6_yes-no-questions_train.txt")

    print("Vocabulary: " + str(len(voc.voc_dict)) + " Words")
    print(str(voc.voc_dict.keys()))

    print("Number of training instances: " + str(len(training_instances)))


if __name__ == "__main__":
    main()
