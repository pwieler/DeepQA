# DeepQA
This project implements approaches for Questions Answering with recurrent neural networks using PyTorch.
The implementation is sepcifically designed to solve four of the the **(20) QA bAbI tasks**:
* bAbI Task 1 - Question anwering with single supporting facts
* bAbI Task 2 - Question answering with two supporting facts
* bAbI Task 3 - Question anwering with three supporting facts
* bAbi Task 6 - Answering of Yes/No Questions

**The datasets used are taken from**
http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz

**For more information on the (20) QA bAbI tasks see here**:
https://research.fb.com/downloads/babi/

**and here**:
https://arxiv.org/abs/1502.05698

## Usage
In the root directory of the repository, main.py implements all necessary steps to train and save a recurrent neural network in order to solve one or multiple bAbI tasks (see section **Multiple bAbI tasks** for information on how to use the network to solve more than one task) .

The log settings and especially the hyper parameters fo the recurrent neural network are done via altering variables at the top of the main() method in main.py.

The paths for the train and test dataset files of the various tasks are preset to having the unzipped dataset located in the folder `./data` under the repo root directory.

It is possible to add more than one value for the hyperparameters. **DeepQA** will then train multiple networks with every possible combination of settings given (for example size of hidden layers, number of hidden layers, learning rate, ...)

There are mutiple implementations for RNN networks, by default, the DeepQA module is used.

### Multiple bAbI tasks
The master branch contains an implementation that is able to train networks on order to solve one bAbI task at a time.
In order to train a network on multiple tasks simultaneously, the branch `multiple_qa_answering` has to be checked out. Multiple QA answering is currently only supported by the DeepQA RNN model.


## Project Structure
.  
├── main.py | *Solves the bAbI QA tasks, parameter are to be set at the begin of main(). Uses QAModel by default.*
├── model | *Contains various RNN implementations for solving the bAbI tasks*
│   ├── QAFFModel.py
│   ├── QAModelLSTM.py
│   ├── QAModel.py | Default RNN implementation using an attention mechanism by combining question and story.
│   ├── SentenceModel.py
│   └── Word2VecEmbedding.py
├── preprocessing | *Contains preprocessing methods to tokenize the bAbI Tasks and interpret them to provide them as a PyTorch Dataset to a PyTorch DataLoader.*
│   └── bAbIData.py
├── README.md
├── results | *Default folder for logging and results as well as trained networks.*
│   └── tmp
└── utils
    └── utils.py