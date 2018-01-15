from typing import List
import re
import torch.nn as nn
import math


class Vocabulary:
    def __init__(self, vocabulary_dict=None, embedding=None):
        self.voc_dict = vocabulary_dict if vocabulary_dict is not None else dict()
        self._embedding = embedding

    def word_to_id(self, word):
        # This is a trick entry for words that are non existent.
        if word is None or word not in self.voc_dict:
            return len(self.voc_dict)

        return self.voc_dict[word]

    def word_to_tensor(self, word):
        if self._embedding is None:
            self._initialize_embedding()

        return self._embedding(self.word_to_id(word))

    def extend_with_word(self, word: str):
        if word not in self.voc_dict: self.voc_dict[word] = len(self.voc_dict)

    def extend_with_text(self, text: str):
        word_set = set()

        for word_blob in text.split():
            alnum_word = "".join([c for c in word_blob if c.isalpha()])
            symbols = "".join([c for c in word_blob if not c.isalnum()])

            if len(alnum_word) > 0:
                if alnum_word not in word_set: word_set.add(alnum_word)

            if len(symbols) > 0:
                if symbols not in word_set: word_set.add(symbols)

        for word in word_set:
            self.extend_with_word(word)

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

    def _initialize_embedding(self, em_dim=-1):
        if em_dim is -1:
            em_dim = math.log2(len(self.voc_dict) + 1)

        self._embedding = nn.Embedding(len(self.voc_dict) + 1, embedding_dim=em_dim)


class TrainingInstance:
    def __init__(self):
        self.indexed_story = []
        self.question = []
        self.answer = []
        self.hints = []

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
        return self.indexed_story[self.hints[0] - 1] + self.indexed_story[self.hints[1] - 1]

    def instances_from_file(path):
        training_instances = []

        with open(path, 'r') as f:
            training_instances += TrainingInstance.instances_from_lines(f.readlines())

            return training_instances

    def instances_from_lines(lines, ):
        training_instances = []

        ind_lines = TrainingInstance._indexed_lines(lines)

        # Either there is no story or the story begins with a question
        if len(ind_lines) is 0 or len(ind_lines[0]) > 2:
            return []

        instance = TrainingInstance()

        current_story = []

        for line in ind_lines:
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

    def _indexed_lines(lines: List[str]):
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


def main():
    voc = Vocabulary()
    voc.extend_with_file("data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt")
    voc.extend_with_file("data/tasks_1-20_v1-2/en/qa2_two-supporting-facts_train.txt")
    voc.extend_with_file("data/tasks_1-20_v1-2/en/qa3_three-supporting-facts_train.txt")
    voc.extend_with_file("data/tasks_1-20_v1-2/en/qa6_yes-no-questions_train.txt")

    training_instances = []
    training_instances += TrainingInstance.instances_from_file(
            "data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt")
    training_instances += TrainingInstance.instances_from_file(
            "data/tasks_1-20_v1-2/en/qa2_two-supporting-facts_train.txt")
    training_instances += TrainingInstance.instances_from_file(
            "data/tasks_1-20_v1-2/en/qa3_three-supporting-facts_train.txt")
    training_instances += TrainingInstance.instances_from_file("data/tasks_1-20_v1-2/en/qa6_yes-no-questions_train.txt")

    print("Vocabulary: " + str(len(voc.voc_dict)) + " Words")
    print(str(voc.voc_dict.keys()))

    print("Number of training instances: " + str(len(training_instances)))


if __name__ == "__main__":
    main()
