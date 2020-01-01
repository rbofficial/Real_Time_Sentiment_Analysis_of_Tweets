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
    tweets = Cursor(api.search, q=search_words, lang="en").items(10000)
    return tweets


auth= authenticate_twitter_app()
print("enter search terms")
str1 = input()
search_words = str1 + ("-filter:retweets")
tweets= search(auth,search_words)

# list that contains text of each tweet
tweet_list = [tweet.text for tweet in tweets]
#print(tweet_list)

# list that removes the url from each tweet of tweet_list
all_tweets_no_url = [remove_url(tweet) for tweet in tweet_list]
#print(all_tweets_no_url)

#list of lists containing lowercase letters for each tweet
words_in_tweet=[tweet.lower().split() for tweet in all_tweets_no_url]
#print(words_in_tweet)


# flatten list of lists into a list using chain
# chain(*iterables)-  example- chain('abc',[1,2,3])-- 'a', 'b', 'c', 1, 2, 3
# Return a chain object whose __next__() method returns elements from the first iterable until it is exhausted,
# then elements from the next iterable, until all of the iterables are exhausted.
all_words_no_urls= list(itertools.chain(*words_in_tweet))
#print(all_words_no_urls)

# for removing stop words
stop_words= set(stopwords.words('english'))

# for all the words in all_words_no_urls , loop checks if
# that word is not present in stop word, then only it will be added to tweets_nsw
tweets_nsw=[word for word in all_words_no_urls if not word in stop_words]
#print(tweets_nsw)

tweets_nsw_lol = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in words_in_tweet]

# for removing collection words.
collection_words= str1.split(" ")
collection_words.append(str1)
tweets_nsw_nc = [[w for w in word if not w in collection_words]
                 for word in tweets_nsw_lol]
final_word_list =[ word for word in tweets_nsw if not word in collection_words]
#print(final_word_list)

                                       #analysing word frequency count using a horizontal bar graph

counts_words= Counter(final_word_list)
# add this data into dataframe
# final_word_list_df  contains 50 most common words of the tweets
final_word_list_df= DataFrame(counts_words.most_common(50),
                             columns=['words', 'count'])
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
# Plot horizontal bar graph
final_word_list_df.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="purple")
ax.set_title("Common Words Found in Tweets (Without Stop or Collection Words)")
#plt.show()


                                    #identifying the co-occurence of words in the tweets and visualising them as networks

terms_bigram = [list(bigrams(tweet)) for tweet in tweets_nsw_nc]
#flatten the list
bigram_list = list(itertools.chain(*terms_bigram))
bigram_counts = Counter(bigram_list)
#contains the 50 most common bigrams
bigram_df = DataFrame(bigram_counts.most_common(50),columns=['bigram','count'])

# visualize the bigrams as network graph
# conversion of DF into a dictionary by to_dict() method
d= bigram_df.set_index('bigram').T.to_dict('records')

# this above method sets 'bigram' as index with count as value - dictionary= {index: value}
# usual to.dict() method sets the column name of the DF as keys/index
# https://stackoverflow.com/questions/26716616/convert-a-pandas-dataframe-to-a-dictionary

G= nx.Graph()

# since bigrams contain two words, this step takes a word and then adds edge with the respective word
for k, v in d[0].items():
    G.add_edge(k[0], k[1], weight=(v * 10))
fig, ax = plt.subplots(figsize=(10, 8))
# for positioning the graph nodes
pos = nx.spring_layout(G, k=1)
# Plot networks
nx.draw_networkx(G, pos,
                 font_size=16,
                 width=3,
                 edge_color='grey',
                 node_color='purple',
                 with_labels=False,
                 ax=ax)
# Create offset labels
for key, value in pos.items():
    x, y = value[0] + .135, value[1] + .045
    ax.text(x, y,
            s=key,
            bbox=dict(facecolor='red', alpha=0.25),
            horizontalalignment='center', fontsize=13)
plt.title("Co-occurrence of words in Tweets")
#plt.show()
                                  # calculate sentiment of each tweet

sentiment_objects = [TextBlob(tweet) for tweet in all_tweets_no_url]
sentiment_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in sentiment_objects]
sentiment_df = DataFrame(sentiment_values, columns=["polarity", "tweet"])
sentiment_df.head()
fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram of the polarity values
sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
             ax=ax,
             color="purple")
plt.title("Overall Sentiment of Tweets")
plt.show()
plt.ioff()
print("its done")
plt.show()

