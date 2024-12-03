# Streamlit Frontend for Twitter Sentiment Analysis

import streamlit as st
import tweepy
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

nltk.download('vader_lexicon')

# Set up Tweepy client
client = tweepy.Client(bearer_token='replace_me')

# Preprocess tweets
def preprocess_tweet(sen):
    sentence = sen.lower()
    sentence = re.sub('RT @\w+: ', " ", sentence)
    sentence = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

# Count values in a column
def count_values_in_column(data, feature):
    total = data.loc[:, feature].value_counts(dropna=False)
    percentage = round(data.loc[:, feature].value_counts(dropna=False, normalize=True) * 100, 2)
    return pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])

# Create wordcloud
def create_wordcloud(text):
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white", max_words=100, stopwords=stopwords)
    wc.generate(str(text))
    return wc

# Streamlit UI
st.title("Twitter Sentiment Analysis")
st.sidebar.header("Configuration")

# Input hashtag or keyword
query = st.sidebar.text_input("Enter a hashtag or keyword:", "#apple")
limit = st.sidebar.slider("Number of pages to fetch (10 tweets per page):", 1, 50, 10)

if st.sidebar.button("Fetch Tweets"):
    with st.spinner("Fetching tweets..."):
        # Fetch tweets using Tweepy
        paginator = tweepy.Paginator(client.search_recent_tweets, query=query + " -is:retweet lang:en",
                                     max_results=10, limit=limit)
        tweet_list = [tweet.text for tweet in paginator.flatten()]
        st.success("Fetched tweets successfully!")

    # Create DataFrame
    tweet_list_df = pd.DataFrame(tweet_list, columns=['text'])
    tweet_list_df['cleaned'] = tweet_list_df['text'].apply(preprocess_tweet)

    # Sentiment Analysis
    tweet_list_df[['polarity', 'subjectivity']] = tweet_list_df['cleaned'].apply(
        lambda Text: pd.Series(TextBlob(Text).sentiment))
    for index, row in tweet_list_df['cleaned'].iteritems():
        score = SentimentIntensityAnalyzer().polarity_scores(row)
        neg, neu, pos, comp = score['neg'], score['neu'], score['pos'], score['compound']
        sentiment = "neutral"
        if comp <= -0.05:
            sentiment = "negative"
        elif comp >= 0.05:
            sentiment = "positive"
        tweet_list_df.loc[index, 'sentiment'] = sentiment
        tweet_list_df.loc[index, 'neg'] = neg
        tweet_list_df.loc[index, 'neu'] = neu
        tweet_list_df.loc[index, 'pos'] = pos
        tweet_list_df.loc[index, 'compound'] = comp

    # Display Sentiment Counts
    st.subheader("Sentiment Distribution")
    sentiment_counts = count_values_in_column(tweet_list_df, "sentiment")
    st.table(sentiment_counts)

    # Pie Chart
    st.subheader("Sentiment Pie Chart")
    fig, ax = plt.subplots()
    names = sentiment_counts.index
    size = sentiment_counts["Percentage"]
    ax.pie(size, labels=names, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    # Wordcloud
    st.subheader("Wordcloud")
    wordcloud = create_wordcloud(tweet_list_df["cleaned"].values)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Display DataFrame
    st.subheader("Fetched Tweets with Sentiment")
    st.dataframe(tweet_list_df)
