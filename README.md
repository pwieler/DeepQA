# DeepQA
Question Answering System with QA bAbI

--> Special branch for testing Sentence Model, because it has a different preprocessing than our standard models!

## Usage

Just run "python DeepQA.py".
For details about the dataset and other prerequisities, please see master-branch.

## Sentence Model
The SentenceModel is a model, where all the facts for one query are read in separateley, instead of reading in all words together!
The idea is that it may perform better, because it can focus more to the facts.

In this model we use three separate GRU-Networks.
First, to read in the query to generate query-encoding.
Second, to read in every fact separately and generate fact-encodings.
Third, to read in the fusioned fact-encodings (fusioned with question-code).

Remember:
Our dataset consists out of stories and queries
--> every story belongs to one query
--> every story consists out of multiple facts
--> one fact is one sentence, e.g.: "Mary is in the house."

On this branch the preprocessing is different in that way, that it generates Datasets in that form:

story: BATCH_SIZE x STORY_MAX_LEN x FACT_MAX_LEN

query: BATCH_SIZE x QUERY_MAX_LEN
