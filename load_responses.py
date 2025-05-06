import pickle
from pprint import pprint

with open("./responses.pickle", "rb") as f:
    responses = pickle.load(f)
for r in responses:
    print(r['response'])