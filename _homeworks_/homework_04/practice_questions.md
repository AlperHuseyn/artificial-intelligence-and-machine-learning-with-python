# 1) Perform multi-class text classification using artificial neural networks with vectorization for the following example:

## Explanations:

- You can download the dataset from the following link: [COVID-19 NLP Text Classification Dataset](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification)

- We will only be concerned with two columns: "OriginalTweet" and "Sentiment." Make sure to read the dataset using Pandas' `read_csv` function with `encoding='latin-1'`.

- The "Sentiment" column consists of categorical data with four categories:
    - Neutral
    - Positive
    - Extremely Negative
    - Negative
    - Extremely Positive

- Note that the output layer of the model should use "softmax." Also, the output data should undergo "one hot encoding" before training. You can directly perform vectorization for the text data using the `CountVectorizer` class since the text can be obtained directly.

- Provide an example of prediction using your own generated text.

# 2) Perform multi-class text classification using artificial neural networks with vectorization for the following example:

## Explanations:

- You can download the dataset from the following link: [Topic Modelling on Emails Dataset](https://www.kaggle.com/datasets/dipankarsrirag/topic-modelling-on-emails)

- The dataset is available as a zip file. Upon extraction, it will create four different directories:
    - Crime
    - Entertainment
    - Politics
    - Science

- There is no separate file for labels in the dataset. The labels are already determined based on the directories in which the emails are located. Inside these directories, there are text files containing the texts that you need to extract and perform vectorization. Also, note that the dataset is not divided into "training" and "test" sets. You need to perform this split yourself. If reading files using the "utf-8" encoding causes issues, try using "latin-1" encoding (default behavior).

- As in the previous example, the output layer of the model should use "softmax."

- Provide an example of prediction using your own generated text.
