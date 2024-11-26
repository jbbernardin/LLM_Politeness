import random
import json
from datasets import load_dataset

squad_dataset = load_dataset("squad_v2")

train_data = list(squad_dataset["train"])

sampled_data = random.sample(train_data, 1000)

with open("sampled_squad_data.json", "w") as json_file:
    json.dump(sampled_data, json_file, indent=4)

print("Sampled dataset saved to 'sampled_squad_data.json'")