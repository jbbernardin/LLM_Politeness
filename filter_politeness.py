import json

# File paths
input_file_path = "politeness.json"
output_file_path = "filtered_politeness.json"

# Load politeness.json
with open(input_file_path, "r") as f:
    politeness_data = json.load(f)

# Filter entries where "answers" is not empty (i.e., "text" and "answer_start" are non-empty)
filtered_data = [
    entry for entry in politeness_data 
    if entry.get("answers", {}).get("text") and entry.get("answers", {}).get("answer_start")
]

# Save the filtered data to a new file
with open(output_file_path, "w") as f:
    json.dump(filtered_data, f, indent=4)

print(f"Filtered data saved to {output_file_path}. Total valid entries: {len(filtered_data)}")
