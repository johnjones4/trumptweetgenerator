from dotenv import load_dotenv
load_dotenv()
from lib import db
from lib.db import Tweet
import csv
from dateutil import parser

db.init_db()

with open('tweets.csv', newline='') as csvfile:
  tweets_csv = csv.reader(csvfile)
  tweets = []
  for tweet_csv in tweets_csv:
    if tweet_csv[4] != "Tweet ID" and tweet_csv[0] is not None and len(tweet_csv[0].strip()) > 0:
      tweets.append(Tweet(
        tweet_id=int(tweet_csv[4]),
        date=parser.parse(tweet_csv[1]),
        text=tweet_csv[0]
      ))
  db.session.add_all(tweets)
  db.session.commit()
  print("Added %d tweets." % (len(tweets), ))