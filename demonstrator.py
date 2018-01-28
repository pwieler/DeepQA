import torch
import argparse
import pickle
import numpy as np
import re
import preprocessing.bAbIData as bd
from model.QAModel import QAModel
from torch.autograd import Variable

hint_c = "\033[1;94m"
normal_c = "\033[39m"


def main():
    parser = argparse.ArgumentParser(description='Use a trained QAModel and do question answering on provided stories.')

    parser.add_argument('model_file', help='The file path of the model weights that are to be used.')
    parser.add_argument('param_file', help='The file path of the parameters that describe this QAModel.')
    parser.add_argument('voc_file', help='The file path of the vocabulary that belongs to this QAModel.')
    parser.add_argument("-c -cuda", help="Use cuda for calculations.", action='store_true', dest="use_cuda")
    parser.add_argument("-v -voc", help="Print the available vocabulary.", action='store_true', dest="print_voc")

    args = parser.parse_args()

    voc, model = load_voc_and_model(args.model_file, args.param_file, args.voc_file, args.use_cuda)

    if args.print_voc:
        print(printable_vocabulary(voc))
        print("\n")

    story = ""

    print(
        "Starting interactive mode. 'q' for exit, 'voc' to show available vocabulary, 'story' to print current story, "
        "'clear' to start new story.")

    while True:
        try:
            line = input("Story line or question: ")
            line = line.strip()

            if line == 'q':
                break

            if line.lower() == "clear":
                story = ""
                print(hint_c + "> Cleared Story. \033[39m")
                continue

            if line.lower() == "story":
                print(hint_c + "> " + story + "\033[39m")
                continue

            if line.lower() == "voc":
                print(hint_c + "> " + printable_vocabulary(voc) + "\033[39m")
                continue

            if line[-1] == '?':
                if len(story) is 0:
                    print(hint_c + "> There is no story to answer the question." + normal_c)
                    continue
                print("\n")
                calculate_and_print(story, line, voc, model, args.use_cuda)
                print("\n")
            else:
                if not check_words(line, voc):
                    continue

                passed, line = check_ending(line)

                if not passed:
                    continue

                story += line + " "

        except EOFError:
            break


def check_words(line, voc):
    unknown_tokens = [word for word in tokenize(line) if voc.word_to_id(word) == voc.unknown_id]

    if len(unknown_tokens) is 0:
        return True

    print(hint_c + "The following words/symbols are not known: " + normal_c)
    print(unknown_tokens)

    reenter = "<answer>"
    while reenter not in "yn":
        reenter = input(hint_c + "Do you want to enter a new sentence instead? ([y]/n) " + normal_c).strip()

    return reenter == 'n'


def check_ending(line):
    if line[-1] is '.':
        return True, line

    reenter = "<answer>"
    while reenter not in "yn":
        reenter = input(
                hint_c + "Your line does not end with a '.' or '?'. Is this part of the story? ([y]/n) " +
                normal_c).strip()

    if reenter.lower() != 'n':
        line += "."
        return True, line
    else:
        return False, line


def printable_vocabulary(voc: bd.Vocabulary):
    return "Vocabulary: " + " | ".join(sorted(list(voc.voc_dict.keys())[2:]))


def calculate_and_print(story, question, voc, model, use_cuda):
    story_toks = tokenize(story)
    question_toks = tokenize(question)

    story_var, question_var, sl_var, ql_var = input_from_story_question(story_toks, question_toks, voc, use_cuda)

    answers = model(story_var, question_var, sl_var, ql_var)

    answers = answers.view(-1)

    top_results = get_most_likely(answers, 3)

    print_results(top_results, voc)


def print_results(top_results, voc: bd.Vocabulary):
    print("{:>10}|{:10}".format("Answer", "Score"))

    for key, value in top_results.items():
        print("{:>10}|{:<10.6f}".format(voc[key.data[0]], np.exp(value)))


def get_most_likely(answers, top_count=5):
    # story_answers = answers.data.cpu().numpy()[0]

    values, indices = answers.sort(0, descending=True)

    results = {}

    for i in range(min(top_count, len(indices))):
        results[indices[i]] = values[i].data[0]

    return results


def tokenize(text):
    tokens = []
    blobs_by_symbols = re.split('([\?\.])',
                                text)  # For example: ['This is a Sentence', '.', ' This is the following Question',
    # '?']

    for blob in blobs_by_symbols:
        blob.strip()
        tokens += blob.split()

    return tokens


def load_voc_and_model(model_file, param_file, voc_file, use_cuda=True):
    voc = bd.Vocabulary()

    with open(voc_file, "rb") as voc_f:
        voc.voc_dict = pickle.load(voc_f)

    with open(param_file, "rb") as param_f:
        param_dict = pickle.load(param_f)

    embedding_size = param_dict["embedding_size"]
    story_hidden_size = param_dict["story_hidden_size"]
    n_layers = param_dict["layers"]
    voc_len = len(voc)

    model = QAModel(voc_len, embedding_size, story_hidden_size, voc_len, n_layers, use_cuda=use_cuda)
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    return voc, model


def input_from_story_question(story, question, voc: bd.Vocabulary, use_cuda=True):
    story = voc.words_to_ids(story)
    question = voc.words_to_ids(question)

    story_tens = torch.LongTensor([story])
    question_tens = torch.LongTensor([question])
    sl_tens = torch.LongTensor([len(story)])
    ql_tens = torch.LongTensor([len(question)])

    story_var = create_variable(story_tens, use_cuda=use_cuda)
    question_var = create_variable(question_tens, use_cuda=use_cuda)
    sl_var = create_variable(sl_tens, use_cuda=use_cuda)
    ql_var = create_variable(ql_tens, use_cuda=use_cuda)

    # Dimensions should be BATCHSIZExSL respective BATCHSIZExQL. should be 1xSL and 1xQL here (1xSL != SL)
    # story_var = story_var.view(1,-1)
    # question_var = question_var.view(1,-1)

    return story_var, question_var, sl_var, ql_var


def create_variable(tensor, use_cuda=True):
    # Do cuda() before wrapping with variable
    if use_cuda and torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


if __name__ == "__main__":
    main()
