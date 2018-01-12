from typing import List
import re
import torch.nn as nn


class LSTMQANet(nn.Module):
    def __init__(self):
        NotImplementedError

    def forward(self):
        NotImplementedError


class Vocabulary:
    def __init__(self, vocabulary_dict, embedding):
        self._vocabulary = vocabulary_dict
        self._embedding = embedding

    def word_to_id(self, word):
        # This is a trick entry for words that are non existent.
        if word is None or word not in self._vocabulary:
            return len(self._vocabulary)

        return self._vocabulary[word]

    def word_to_tensor(self, word):
        return self._embedding(self.word_to_id(word))

    def words_to_tensors(self, words):
        tensors = []

        for word in words:
            tensors.append(self.word_to_tensor())

        return tensors


class TrainingInstance:
    def __init__(self):
        self.indexed_story = []
        self.question = []
        self.answer = []
        self.hints = []

    def story(self):
        flat_story = []

        for sentence in self.indexed_story:
            flat_story += sentence[1]

        return flat_story

    def question(self):
        return self.question

    def answer(self):
        return self.answer

    def hints(self):
        self.indexed_story[self.hints[0] - 1] + self.indexed_story[self.hints[1] - 1]


def extract_stories(lines: List[str]):
    """
    Tokenize the stories and their information for easy access.

    :param lines: The lines of a QA babi file.

    :return: Returns a list of lists which each contain the story line number,
                a list of the tokenized words in the sentence and in those lines where it applies,
                the answer to the question and the hint line numbers.

                Example: [[9, ['Daniel', 'went', 'back', 'to', 'the', 'kitchen', '.']], [10, ['Where', 'is', 'the', 'football', '?'], 'kitchen', [6, 9]]]
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


def training_instances_from_indexed_lines(lines):
    training_instances = []

    # Either there is no story or the story begins with a question
    if len(lines) is 0 or len(lines[0]) > 2:
        return []

    instance = TrainingInstance()

    current_story = []

    for line in lines:
        if line[0] is 1:
            instance = TrainingInstance()
            current_story = []

        if len(line) < 3:
            current_story.append((line[0], line[1]))
        else:
            instance.indexed_story = current_story
            instance.question = line[1]
            instance.answer = line[2]
            instance.hints = line[3]

            training_instances.append(instance)

    return training_instances


def file_to_stories(path="data/qa2_two-supporting-facts_train.txt"):
    with open(path, 'r') as f:
        return extract_stories(f.readlines())


# def token_to_tensor(stories, vocabulary, embedding):


def train(net:LSTMQANet, training_instances: List[TrainingInstance], vocabulary : Vocabulary):
    for epoch in range(10):
        for instance in training_instances:

            story_input = vocabulary.words_to_tensors(instance.story())

            # TODO

            net.forward()


def main():
    print(str(training_instances_from_indexed_lines(file_to_stories())))


if __name__ == '__main__':
    main()
