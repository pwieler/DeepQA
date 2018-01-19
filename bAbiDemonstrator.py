import torch
from model import QAModel
from preprocessing import bAbIData as bd



def main():
    PREVIOUSLY_TRAINED_MODEL="results/2018_01_16_21_58_11_100_100_100_1_32_40_22_0.001_task_1/trained_model.pth"

    EMBED_HIDDEN_SIZE=100
    STORY_HIDDEN_SIZE=100
    QUERY_HIDDEN_SIZE=100
    N_LAYERS=1
    BATCH_SIZE=32
    EPOCHS=40
    VOC_SIZE=22
    LEARNING_RATE=0.001000

    model = QAModel(VOC_SIZE, EMBED_HIDDEN_SIZE, STORY_HIDDEN_SIZE, QUERY_HIDDEN_SIZE, VOC_SIZE, N_LAYERS)
    model.load_state_dict(torch.load(PREVIOUSLY_TRAINED_MODEL))

    #bd.BAbIInstance.

if __name__ == "__main__":
    main()