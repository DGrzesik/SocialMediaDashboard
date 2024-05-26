import numpy as np
from scipy.special import expit
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

MODEL = f"cardiffnlp/tweet-topic-latest-multi"
tokenizer = AutoTokenizer.from_pretrained(MODEL)


def get_topic(text):
    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    class_mapping = model.config.id2label

    # text = "It is great to see athletes promoting awareness for climate change."
    tokens = tokenizer(text, return_tensors='pt')
    output = model(**tokens)

    scores = output[0][0].detach().numpy()
    scores = expit(scores)

    # Find and return the most accurate topic
    max_index = np.argmax(scores)
    return class_mapping[max_index]
