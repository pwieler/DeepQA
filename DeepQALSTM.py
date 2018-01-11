from typing import List
import re


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
            sentence = splitter[0] + splitter[1] # The sentence including questionmark

            answer = splitter[2].split("\t")[1]
            supporting_facts = [int(st) for st in splitter[2].split("\t")[2].split()]

            line[1] = sentence
            line.append(answer)
            line.append(supporting_facts)

    for line in indexed_lines:
        line[1] = [st for st in re.split("(\W)", line[1]) if st not in ["", " ", "\n"]]

    return indexed_lines


def story_from_indexed_lines(lines):

    stories = []

    # Either there is no story or the story begins with a question
    if len(lines) is 0 or len(lines[0]) > 2:
        return []

    current_story = []
    current_story_line = 0
    current_question = []
    current_answer = None
    current_hints = []

    for line in lines:
        current_story is [] if line[0] is 0 else current_story

        if len(line)< 3:
            current_story += line[1]
        else:
            current_question = line[1]
            current_answer = line[2]
            current_hints = line[3]

            stories.append([current_story, current_question, current_answer, current_hints])
            current_question = []
            current_answer = None
            current_hints = []

    return stories




def file_to_stories(path="data/qa2_two-supporting-facts_train.txt"):
    with open(path, 'r') as f:
        return extract_stories(f.readlines())

#def token_to_tensor(stories, vocabulary, embedding):


def main():
    print(str(story_from_indexed_lines(file_to_stories())))


if __name__ == '__main__':
    main()
