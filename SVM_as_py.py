import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
from nltk.corpus import twitter_samples
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud

#data_ex = pd.read_csv(r'')
nltk.download('twitter_samples') # includes 5000 positive, 5000 negative tweets

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

tweets = positive_tweets + negative_tweets

print(tweets[:10], tweets[-10:-1])


polarities = np.append(np.ones((len(positive_tweets))), np.zeros((len(negative_tweets))))

# split --> 20/80 (1000 tweets / 4000 tweets)

testset_pos = positive_tweets[4000:]
testset_neg = negative_tweets[4000:]

trainset_pos = positive_tweets[:4000]
trainset_neg = negative_tweets[:4000]

train_x = trainset_pos + trainset_neg 
test_x = testset_pos + testset_neg

train_y = np.append(np.ones((len(trainset_pos), 1)), np.zeros((len(trainset_neg), 1)), axis=0)
test_y = np.append(np.ones((len(testset_pos), 1)), np.zeros((len(testset_neg), 1)), axis=0)
print(train_y.shape)
print(test_y.shape)

# overview of the data:

train_y_df = pd.DataFrame(train_y)
test_y_df = pd.DataFrame(test_y)

print(train_y_df[:5], test_y_df[:5])

trainset_pos_df = pd.DataFrame(trainset_pos)
print(trainset_pos_df)

def process_tweet(tweet):
   
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    tweet = re.sub(r'\$\w*', '', tweet) # weird tickers
    tweet = re.sub(r'^RT[\s]+', '', tweet) # Retweets
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet) # hyperlinks
    tweet = re.sub(r'#', '', tweet) # remove hashtags
    
    # tokenize
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                              reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punct
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean

def build_freqs(tweets, sentiment_label):
    
    # tweets: a list of tweets
    # sentiment_label: an m x 1 array with the sentiment label of each tweet (either 0 or 1)
   
    sentiment_label_list = np.squeeze(sentiment_label).tolist()

    freqs = {}
    for y, tweet in zip(sentiment_label_list, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    
    # freqs: a dictionary mapping each (word, sentiment) pair to its frequency
    
    return freqs

freqs = build_freqs(tweets, polarities)

# check data type
print(f'type(freqs) = {type(freqs)}')

# check length of the dictionary
print(f'len(freqs) = {len(freqs)}')

print('This is an example of a positive tweet: \n', train_x[0])
print('\nThis is an example of the processed version of the tweet: \n', process_tweet(train_x[0]))
list_of_all = []

svm = svm(kernel = 'linear', probability=True)

# fit the SVC model based on the given training data
prob = svm.fit(train_x, train_y).predict_proba(test_x)

# perform classification and prediction on samples in x_test
y_pred_svm = svm.predict(test_x)

print("Accuracy score for SVC is: ", accuracy_score(test_y, y_pred_svm) * 100, '%')