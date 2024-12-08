import json
import openai
import os
from tqdm import tqdm

input_file_path = "filtered_politeness_modified.json"
output_file_path = "4o-mini_answers.json"

# Load the dataset
with open(input_file_path, "r") as f:
    dataset = json.load(f)

# Set up OpenAI API credentials
openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.api_base = "https://cmu.litellm.ai"
openai.api_type = "open_ai"

def get_answer(context, question):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Please provide a concise and accurate "
                    "answer to the following question. Keep your answer under 20 words."
                ),
            },
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"},
        ],
        temperature=0,
        max_tokens=25,  # Adjust max_tokens to limit the length of the response
    )
    return response["choices"][0]["message"]["content"].strip()

try:
    # Iterate through the dataset
    for idx, entry in enumerate(tqdm(dataset, desc="Processing entries")):
        context = entry["context"]
        ground_truth_answer = entry["answers"]["text"][0]

        # get baseline question answer
        entry["question_answer"] = get_answer(context, entry["question"])

        # Add the ground truth answer
        entry["ground_truth_answer"] = ground_truth_answer

        # Get answers for each question type
        for i in range(1, 4):
            entry[f"neutral_answer_{i}"] = get_answer(context, entry[f"neutral_question_{i}"])
            entry[f"polite_answer_{i}"] = get_answer(context, entry[f"polite_question_{i}"])
            entry[f"impolite_answer_{i}"] = get_answer(context, entry[f"impolite_question_{i}"])

except Exception as e:
    print(f"Error processing entry {idx}: {e}")
    print("Saving progress and exiting...")

# Save the updated dataset to file
with open(output_file_path, "w") as f:
    json.dump(dataset, f, indent=4)

print(f"Processing complete. Updated dataset saved to {output_file_path}")