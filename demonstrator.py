import torch
import pickle
import operator
import preprocessing.bAbIData as bd
from model.QAModel import QAModel
from torch.autograd import Variable


def main():
    model_file = "results/2018_01_27_17_12_55_50_200_2_32_40_40_0.0001_40_tasks_qa1_qa6/trained_model.pth"
    param_file = "results/2018_01_27_17_12_55_50_200_2_32_40_40_0.0001_40_tasks_qa1_qa6/params.pkl"
    voc_file = "results/2018_01_27_17_12_55_50_200_2_32_40_40_0.0001_40_tasks_qa1_qa6/vocabulary.pkl"

    use_cuda = False

    voc, model = load_voc_and_model(model_file, param_file, voc_file, use_cuda)

    test_story = "John moved to the hallway . John journeyed to the kitchen .".split()
    test_question = "Where is John ?".split()

    story_var, question_var, sl_var, ql_var = input_from_story_question(test_story, test_question, voc, use_cuda)

    answer_id = model(story_var, question_var, sl_var, ql_var)

    max_index, max_value = max(enumerate(answer_id.data[0]), key=operator.itemgetter(1))

    print(max_index)

    print(voc)


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
