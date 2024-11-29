import json
import openai 

input_file_path = "sampled_squad_data.json"
output_file_path = "updated_politeness.json"

with open(input_file_path, "r") as f:
    dataset = json.load(f)

openai.api_key = "your_api_key"
openai.api_base = "https://cmu.litellm.ai"
openai.api_type = "open_ai"

# Iterate through the dataset
for idx, entry in enumerate(dataset):
    if "polite_question" in entry and "impolite_question" in entry:
        # Skip if already processed
        print(f"Skipping entry {idx}: Already processed.")
        continue

    original_question = entry["question"]
    print(f"Processing entry {idx}: {original_question}")

    try:
        # Make the OpenAI API call
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"""
                    Generate a polite and impolite version of this question: '{original_question}'. This should be in the format of 'Polite: ' and 'Impolite: ' for the polite and impolite questions respectively. The impolite question must be rude and terse, such that its syntax is notably different from the original question. The polite question can be of the format "Could you please..." or "Could you kindly...", but I'd like if you would use different variations like "May you please..." or "Would you please..." as well. Here are some examples:

                    Example Question 1: Who was the first president of the United States?
                    Polite: May you please tell me who the first president of the United States was?
                    Impolite: Tell me who the first president of the United States is right now.

                    Example Question 2: What caused World War I?
                    Polite: Would you please tell me what caused World War I?
                    Impolite: I need you to tell me what caused World War I.

                    Example Question 3: When did the Berlin Wall fall?
                    Polite: Please tell me when the Berlin Wall fell.
                    Impolite: Tell me when the Berlin Wall fell.
                    """
                },
            ],
            temperature=0.5,
        )

        # Extract polite and impolite questions
        polite_question = response["choices"][0]["message"]["content"].split("Polite: ")[-1].split("\n")[0]
        impolite_question = response["choices"][0]["message"]["content"].split("Impolite: ")[-1].strip()

        # Update the entry with the new fields
        entry["polite_question"] = polite_question
        entry["impolite_question"] = impolite_question

        # Save the updated dataset to file
        with open(output_file_path, "w") as f:
            json.dump(dataset, f, indent=4)

        print(f"Entry {idx} processed and saved.")
    
    except Exception as e:
        print(f"Error processing entry {idx}, {original_question}: {e}")
        print("Saving progress and exiting...")
        # Save progress before exiting
        with open(output_file_path, "w") as f:
            json.dump(dataset, f, indent=4)
        break

print(f"Processing complete. Updated dataset saved to {output_file_path}")