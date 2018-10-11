from dotenv import load_dotenv
load_dotenv()
from lib import db
from lib.db import Tweet
import twitter
from os import getenv
from dateutil import parser
import json

print("Starting up")

db.init_db()

api = twitter.Api(consumer_key=getenv("TWITTER_CONSUMER_KEY"),
                  consumer_secret=getenv("TWITTER_CONSUMER_SECRET"),
                  access_token_key=getenv("TWITTER_ACCESS_TOKEN_KEY"),
                  access_token_secret=getenv("TWITTER_ACCESS_TOKEN_SECRET"))

load_more = True
last_id = None
newest_tweet = db.session.query(Tweet).order_by(Tweet.date.desc()).limit(1).one_or_none()
while load_more:
  statuses = api.GetUserTimeline(
    screen_name="RealDonaldTrump",
    max_id=last_id,
    since_id=newest_tweet.tweet_id if newest_tweet is not None else None,
    count=200,
    include_rts=False,
    trim_user=True
  )
  tweets = []
  for status in statuses:
    tweet = db.session.query(Tweet).filter(Tweet.tweet_id==status.id).limit(1).one_or_none()
    if tweet is None:
      tweets.append(Tweet(
        tweet_id=status.id,
        date=parser.parse(status.created_at),
        text=status.text
      ))
  load_more = len(statuses) > 0
  if len(statuses) > 0:
    last_id = statuses[len(statuses) - 1].id - 1
  db.session.add_all(tweets)
  db.session.commit()
  print("Added %d tweets." % (len(tweets), ))

def load_tweets():
  print('test')

