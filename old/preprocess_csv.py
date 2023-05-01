import pandas as pd
import spacy
from textblob import TextBlob

# TextBlob:: pretty bad model, should use a model trained for something related to the stock market to label the tweets


def classify_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity == 0: return 0
    elif polarity <= 0: return -1
    else: return 1


def classify_sentiment_spacy(text):
    nlp = spacy.load("en_core_web_sm")
    """
    non-functional
    """
    doc = nlp(text)
    if not doc:
        return -1  # Invalid text, return a negative value to indicate error
    if doc.sentiment:
        polarity = doc.sentiment.polarity
        if polarity <= 0:
            return 0  # Negative sentiment
        else:
            return 1  # Positive sentiment
    else:
        return -1  # Sentiment analysis failed, return a negative  value to indicate error


df_original = pd.read_csv('stock_tweets.csv')[:200]
df_new = pd.DataFrame({'tweet_id': range(len(df_original)),
                       'sentiment': df_original['Tweet'].apply(classify_sentiment),
                       'tweet': df_original['Tweet']})

df_new.to_csv('correct_form2.csv', index=False)
