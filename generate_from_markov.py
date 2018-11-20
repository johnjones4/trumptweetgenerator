from dotenv import load_dotenv
load_dotenv()
from lib.markov import MarkovChain
import os

markovchain = MarkovChain()
markovchain.import_chain(os.getenv('MARKOV_CHAIN_FILE','./data/markov.json'))
print(markovchain.generate())
