import itertools
# need to use nltk downloader using console
from nltk.corpus import stopwords
from tweepy import Cursor, API, OAuthHandler
from nltk import bigrams
import re
import twitter_credentials
from collections import Counter
from pandas import DataFrame
import matplotlib.pyplot as plt

import networkx as nx
from textblob import TextBlob

# to reomve url from each tweet
def remove_url(tweet):
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", tweet).split())

def authenticate_twitter_app():
    auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
    auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
    return auth

# function to search for tweets containing the search_words
def search(auth,search_words):
    api = API(auth, wait_on_rate_limit=True)
    # will return first 10,000 tweets
    tweets = Cursor(api.search, q=search_words, lang="en").items(5000)
    return tweets


auth= authenticate_twitter_app()
print("enter search terms")
str1 = input()
search_words = str1 + ("-filter:retweets")
tweets= search(auth,search_words)
tweet_list = [tweet.text for tweet in tweets]

# list that removes the url from each tweet of tweet_list
all_tweets_no_url = [remove_url(tweet) for tweet in tweet_list]

# calculating the sentiment value of each tweet
sentiment_objects = [TextBlob(tweet) for tweet in all_tweets_no_url]
sentiment_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in sentiment_objects]
sentiment_df = DataFrame(sentiment_values, columns=["polarity", "tweet"])
sentiment_df.head()
figure, axi = plt.subplots(figsize=(8, 6))

# Plot histogram of the polarity values
sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
             ax=axi,
             color="purple")
plt.title("Overall Sentiment of Tweets")


dict_sv= {}
for tweet in sentiment_objects:
    dict_sv.update({str(tweet) : tweet.sentiment.polarity})

# making a list of tweets that have positive sentiments
pos_tweet_list= []
for i in dict_sv:
    if dict_sv[i] >= 0.0:
        pos_tweet_list.append(i)

# making a list of tweets that have negative sentiment
neg_tweet_list=[]
for i in dict_sv:
    if dict_sv[i] < 0.0:
        neg_tweet_list.append(i)

#list of lists containing lowercase letters for each tweet
pos_words_in_tweet=[tweet.lower().split() for tweet in pos_tweet_list]
neg_words_in_tweet=[tweet.lower().split() for tweet in neg_tweet_list]


# flatten list of lists into a list using chain
# chain(*iterables)-  example- chain('abc',[1,2,3])-- 'a', 'b', 'c', 1, 2, 3
# Return a chain object whose __next__() method returns elements from the first iterable until it is exhausted,
# then elements from the next iterable, until all of the iterables are exhausted.
pos_all_words_no_urls= list(itertools.chain(*pos_words_in_tweet))
neg_all_words_no_urls= list(itertools.chain(*neg_words_in_tweet))

# for removing stop words
stop_words= set(stopwords.words('english'))

# for all the words in all_words_no_urls , loop checks if
# that word is not present in stop word, then only it will be added to tweets_nsw
pos_tweets_nsw=[word for word in pos_all_words_no_urls if not word in stop_words]
neg_tweets_nsw=[word for word in neg_all_words_no_urls if not word in stop_words]

pos_tweets_nsw_lol = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in pos_words_in_tweet]

neg_tweets_nsw_lol = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in neg_words_in_tweet]


# for removing collection words
collection_words= str1.split(" ")
collection_words.append(str1)
pos_tweets_nsw_nc = [[w for w in word if not w in collection_words]
                 for word in pos_tweets_nsw_lol]
pos_final_word_list =[ word for word in pos_tweets_nsw if not word in collection_words]

neg_tweets_nsw_nc = [[w for w in word if not w in collection_words]
                 for word in neg_tweets_nsw_lol]
neg_final_word_list =[ word for word in neg_tweets_nsw if not word in collection_words]


pos_counts_words= Counter(pos_final_word_list)
neg_counts_words= Counter(neg_final_word_list)

# final_word_list_df  contains 50 most common words of the tweets
pos_final_word_list_df= DataFrame(pos_counts_words.most_common(50),
                             columns=['words', 'count'])
neg_final_word_list_df= DataFrame(neg_counts_words.most_common(50),
                             columns=['words', 'count'])
plt.ion()

                                        # analyzing the word frequency count

# plotting a horizontal bar graph for positive tweets
fig, ax = plt.subplots(figsize=(8, 8))
pos_final_word_list_df.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="purple")
ax.set_title("Common Words Found in Positive Tweets (Without Stop or Collection Words)")

# plotting a horizontal bar graph for negative tweets
fig, ax1 = plt.subplots(figsize=(8, 8))
neg_final_word_list_df.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax1,
                      color="purple")
ax1.set_title("Common Words Found in Negative Tweets (Without Stop or Collection Words)")


                            # ananlysing the co-occurence of words (using bigrams)


pos_terms_bigram = [list(bigrams(tweet)) for tweet in pos_tweets_nsw_nc]
neg_terms_bigram = [list(bigrams(tweet)) for tweet in neg_tweets_nsw_nc]
#flatten the list
pos_bigram_list = list(itertools.chain(*pos_terms_bigram))
pos_bigram_counts = Counter(pos_bigram_list)
#contains the 50 most common bigrams
pos_bigram_df = DataFrame(pos_bigram_counts.most_common(40),columns=['bigram','count'])

neg_bigram_list = list(itertools.chain(*neg_terms_bigram))
neg_bigram_counts = Counter(neg_bigram_list)
#contains the 50 most common bigrams
neg_bigram_df = DataFrame(neg_bigram_counts.most_common(40),columns=['bigram','count'])


# conversion of DF into a dictionary by to_dict() method
# this above method sets 'bigram' as index with count as value - dictionary= {index: value}
# usual to.dict() method sets the column name of the DF as keys/index
# https://stackoverflow.com/questions/26716616/convert-a-pandas-dataframe-to-a-dictionary
pos_d= pos_bigram_df.set_index('bigram').T.to_dict('records')
neg_d= neg_bigram_df.set_index('bigram').T.to_dict('records')


pos_G= nx.Graph()
# since bigrams contain two words, this step takes a word and then adds edge with the respective word
for k, v in pos_d[0].items():
    pos_G.add_edge(k[0], k[1], weight=(v * 10))
fig1, ax1 = plt.subplots(figsize=(10, 8))
# for positioning the graph nodes
pos = nx.spring_layout(pos_G, k=1)
# Plot networks
nx.draw_networkx(pos_G, pos,
                 font_size=10,
                 width=3,
                 edge_color='grey',
                 node_color='purple',
                 with_labels=False,
                 ax=ax1)
# Create offset labels
for key, value in pos.items():
    x, y = value[0] + .135, value[1] + .045
    ax1.text(x, y,
            s=key,
            bbox=dict(facecolor='red', alpha=0.25),
            horizontalalignment='center', fontsize=13)
plt.title("Co-occurrence of words in positive Tweets")

neg_G= nx.Graph()
# since bigrams contain two words, this step takes a word and then adds edge with the respective word
for k, v in neg_d[0].items():
    neg_G.add_edge(k[0], k[1], weight=(v * 10))
fig2, ax2 = plt.subplots(figsize=(10, 8))
# for positioning the graph nodes
posi = nx.spring_layout(neg_G, k=1)
# Plot networks
nx.draw_networkx(neg_G, posi,
                 font_size=10,
                 width=3,
                 edge_color='grey',
                 node_color='purple',
                 with_labels=False,
                 ax=ax2)
# Create offset labels
for key, value in posi.items():
    x, y = value[0] + .135, value[1] + .045
    ax2.text(x, y,
            s=key,
            bbox=dict(facecolor='red', alpha=0.25),
            horizontalalignment='center', fontsize=13)
plt.title("Co-occurrence of words in Negative Tweets")
plt.show()
plt.ioff()
print("its done")
plt.show()
