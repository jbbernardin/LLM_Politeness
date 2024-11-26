import json
import openai 

file_path = "sampled_squad_data.json"

with open(file_path, "r") as f:
    dataset = json.load(f)

openai.api_key = "your_key_here"
openai.api_base = "https://cmu.litellm.ai"
openai.api_type = "open_ai"

for entry in dataset:
    original_question = entry["question"]

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Generate a polite and impolite version of this question: '{original_question}'. This should be in the format of 'Polite: ' and 'Impolite: ' for the polite and impolite questions respectively.",
            },
        ],
    )

    polite_question = response["choices"][0]["message"]["content"].split("Polite: ")[-1].split("\n")[0]
    impolite_question = response["choices"][0]["message"]["content"].split("Impolite: ")[-1].strip()

    entry["polite_question"] = polite_question
    entry["impolite_question"] = impolite_question

output_file = "updated_squad_data.json"
with open(output_file, "w") as f:
    json.dump(dataset, f, indent=4)

print(f"Updated dataset saved to {output_file}")
