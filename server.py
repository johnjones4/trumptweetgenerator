from dotenv import load_dotenv
load_dotenv()
from lib.markov import MarkovChain
import os
from flask import Flask, jsonify

app = Flask(__name__)

markovchain = MarkovChain()
markovchain.import_chain(os.getenv('MARKOV_CHAIN_FILE','./data/markov.json'))

@app.route("/markov")
def hello():
  return jsonify({"tweet": markovchain.generate()})

if __name__ == '__main__':
  app.run()
