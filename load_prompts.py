import pickle

with open("./prompts.pickle", "rb") as f:
    prompts = pickle.load(f)

print(len(prompts))