from dotenv import load_dotenv
load_dotenv()
from lib import db
from lib.db import Tweet
from lib.markov import MarkovChain
import os
import re

db.init_db()
tweets_objects = db.session.query(Tweet).order_by(Tweet.date.desc()).limit(200)
regex = re.compile(r"((http[s]?|ftp):\/)?\/?([^:\/\s]+)((\/\w+)*\/)([\w\-\.]+[^#?\s]+)(.*)?(#[\w\-]+)?", re.IGNORECASE)
tweet_list = list(map(lambda tweet: regex.sub("", tweet.text), tweets_objects))

markovchain = MarkovChain()
markovchain.load_texts(tweet_list)
markovchain.export_chain(os.getenv('MARKOV_CHAIN_FILE','./data/markov.json'))
