import pickle
from pprint import pprint

with open("./prompts.pickle", "rb") as f:
    prompts = pickle.load(f)
print(''.join(list(prompts[0])))

