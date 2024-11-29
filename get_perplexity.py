import os
import math
import openai
import json
from tqdm import tqdm

# Function to calculate perplexity from log probabilities
def calculate_perplexity(log_probs):
    # Filter out None values from log_probs
    log_probs = [lp for lp in log_probs if lp is not None]
    avg_log_prob = sum(log_probs) / len(log_probs)
    return math.exp(-avg_log_prob)

# Initialize the LiteLLM client
client = openai.OpenAI(
    api_key="API_KEY",
    base_url="https://cmu.litellm.ai",
)

# Read the JSON file
with open('filtered_politeness.json', 'r') as file:
    questions_data = json.load(file)

# Prepare the output data
output_data = []

try:
    # Process each question
    for entry in tqdm(questions_data, desc="Processing questions"):
        data_point = {}
        for key in ["question", "polite_question", "impolite_question"]:
            question = entry[key]

            # Make a request to a model that supports logprobs (e.g., text-davinci-003)
            response = client.completions.create(
                model="davinci-002",  # Ensure this model supports logprobs
                prompt=question,
                max_tokens=0,  # We only want logprobs for the prompt
                logprobs=1,  # Enable log probabilities
                echo=True  # Include input tokens in the response
            )

            # Extract tokens and log probabilities from the response
            choices = response.choices  # Access choices attribute of the response object
            tokens = choices[0].logprobs.tokens  # Tokens from logprobs
            log_probs = choices[0].logprobs.token_logprobs  # Log probabilities for tokens

            # Calculate perplexity
            perplexity = calculate_perplexity(log_probs)

            # Append results to output data
            data_point[key] = ({
                "question": question,
                "tokens": tokens,
                "log_probs": log_probs,
                "perplexity": perplexity
            })
        # Append the data point to the output data
        output_data.append(data_point)

    # Write the output data to a new file
    with open('perplexity_results.json', 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

except Exception as e:
    print(f"An error occurred: {e}")
    with open('perplexity_results.json', 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

print("Perplexity evaluation completed and results saved to perplexity_results.json")