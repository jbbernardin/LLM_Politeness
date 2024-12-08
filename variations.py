import openai
import json
import time
import re

# Set up OpenAI API credentials
openai.api_key = "your_api_key  "
openai.api_base = "https://cmu.litellm.ai"
openai.api_type = "open_ai"

last_processed_index = -1

def generate_variations_via_openai(base_question, count, tone="neutral"):
    """
    Generate variations of a base question using OpenAI API and the custom prompt.
    """
    prompt = (
        f"I have a sentence, and I want to generate {count} variations of it. "
        "Each variation should retain the original tone, meaning, and intent but use different phrasing or structure. "
        "Please return the output as a JSON array of strings. For example:\n\n"
        "Input: 'What is the best way to learn programming?'\n"
        "Output: [\n"
        "  'How can I effectively learn programming?',\n"
        "  'What are the most effective methods to learn programming?'\n"
        "]\n\n"
        f"Now, generate {count} {tone} variations for this sentence:\n'{base_question}'"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant generating {tone} variations of questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=150
        )
        # Extract and clean generated variations
        content = response['choices'][0]['message']['content'].strip()
        print(content)  # Debugging: Print raw output for inspection

        # Extract JSON content from code block if present
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
        if json_match:
            content = json_match.group(1)

        variations = json.loads(content)
        print(variations)
        while len(variations) < count:
            variations.append(f"Placeholder for {tone} question {len(variations) + 1}")
        return variations[:count]
    except openai.error.RateLimitError:
        print("Rate limit exceeded. Retrying after 10 seconds...")
        time.sleep(10)
        return generate_variations_via_openai(base_question, count, tone)
    # except openai.error.OpenAIError as e:
    #     print(f"OpenAI API error: {e}")
    #     return [f"Placeholder for {tone} question {i+1}" for i in range(count)]
    # except Exception as e:
    #     print(f"Unexpected error: {e}")
    #     return [f"Placeholder for {tone} question {i+1}" for i in range(count)]

# Load the original JSON
with open('filtered_politeness.json', 'r') as file:
    data = json.load(file)

# Process each entry in the JSON
for idx, entry in enumerate(data):
    if idx <= last_processed_index:
        continue
    print(f"Processing entry {idx}: {entry['question']}")
    
    try:
        neutral_variations = generate_variations_via_openai(entry['question'], count=3, tone="neutral")
        entry['neutral_question_1'] = neutral_variations[0]
        entry['neutral_question_2'] = neutral_variations[1]
        entry['neutral_question_3'] = neutral_variations[2]

        # Generate polite question variations
        polite_base = entry.pop('polite_question', None)
        polite_variations = generate_variations_via_openai(polite_base, count=2, tone="polite")
        entry['polite_question_1'] = polite_base
        entry['polite_question_2'] = polite_variations[0]
        entry['polite_question_3'] = polite_variations[1]

        # Generate impolite question variations
        impolite_base = entry.pop('impolite_question', None)
        impolite_variations = generate_variations_via_openai(impolite_base, count=2, tone="impolite")
        entry['impolite_question_1'] = impolite_base
        entry['impolite_question_2'] = impolite_variations[0]
        entry['impolite_question_3'] = impolite_variations[1]

        # Save the modified JSON to a new file after processing each entry
        with open('filtered_politeness_modified.json', 'w') as file:
            json.dump(data, file, indent=4)

    except Exception as e:
        print(f"Error processing entry {idx}: {e}")
        continue
        
