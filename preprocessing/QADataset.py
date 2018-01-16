from torch.utils.data import Dataset

class QADataset(Dataset):

    def __init__(self, story, query, answer, story_lengths, query_lengths):
        self.story = story
        self.query = query
        self.answer = answer
        self.story_lengths = story_lengths
        self.query_lengths = query_lengths
        self.len = len(story)

    def __getitem__(self, index):
        return self.story[index], self.query[index], self.answer[index], self.story_lengths[index], self.query_lengths[index]

    def __len__(self):
        return self.len