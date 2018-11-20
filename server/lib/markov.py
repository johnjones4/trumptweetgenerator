import json
import random
import re

STOP = "STOP"

regex = re.compile(r"(\n|\r|\.|\,|\;|\'|\"|\ |\(|\))+", re.IGNORECASE)

class MarkovChain:
  chain = {}
  start_chain = {}

  def load_texts(self, texts):
    temp_chain = {}
    temp_start_chain = {}
    for text in texts:
      words = list(map(lambda word: regex.sub("", word), text.lower().split(" ")))
      if len(words) > 0:
        first_word = words[0]
        if first_word not in temp_start_chain:
          temp_start_chain[first_word] = 1
        else:
          temp_start_chain[first_word] += 1

        for (index, word) in enumerate(words):
          if word not in temp_chain:
            temp_chain[word] = {}
          next_word = words[index + 1] if index + 1 < len(words) else STOP
          if next_word not in temp_chain[word]:
            temp_chain[word][next_word] = 1
          else:
            temp_chain[word][next_word] += 1

    new_chain = {}
    for word in temp_chain:
      total = 0
      for next_word in temp_chain[word]:
        total += temp_chain[word][next_word]
      new_chain[word] = {}
      smallest_pcnt = 1
      for next_word in temp_chain[word]:
        pcnt = float(temp_chain[word][next_word]) / float(total)
        new_chain[word][next_word] = pcnt
        if pcnt < smallest_pcnt:
          smallest_pcnt = pcnt
      factor = 1.0 / smallest_pcnt
      for next_word in temp_chain[word]:
        new_chain[word][next_word] = int(new_chain[word][next_word] * factor)

    new_start_chain = {}
    smallest_pcnt = 1
    for word in temp_start_chain:
      pcnt = float(temp_start_chain[word]) / float(len(temp_start_chain))
      new_start_chain[word] = pcnt
      if pcnt < smallest_pcnt:
          smallest_pcnt = pcnt
        
    factor = 1.0 / smallest_pcnt
    for word in temp_start_chain:
      new_start_chain[word] = int(new_start_chain[word] * factor)

    self.chain = new_chain
    self.start_chain = new_start_chain

  def export_chain(self, filepath):
    json_str = json.dumps({
      "chain": self.chain,
      "start_chain": self.start_chain
    })
    with open(filepath, "w") as export_file:
      export_file.write(json_str)

  def import_chain(self, filepath):
    with open(filepath, "r") as import_file:
      json_str = import_file.read()
      imported = json.loads(json_str)
      self.chain = imported["chain"]
      self.start_chain = imported["start_chain"]

  def generate(self):
    last_word = None
    sentence = ""
    while True:
      word_dict = None
      if len(sentence) == 0:
        word_dict = self.start_chain
      elif last_word in self.chain:
        word_dict = self.chain[last_word]
      else:
        return sentence
      words_array = []
      for word in word_dict:
        words_array += [word] * (word_dict[word])
      next_word = random.choice(words_array)
      if next_word == STOP:
        return sentence
      else:
        if len(sentence) > 0:
          sentence += " "
        sentence += next_word
        last_word = next_word
      
