import os

# Input files (replace with your actual file paths if different)
file_paths = [
    "combined_csv_data.txt",
    "combined_json_data.txt",
    "combined_text_data.txt",
    "combined_web_data.txt"
]

# Output file
output_file = "all_combined_tickets.txt"

def combine_files(input_files, output_file):
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file_path in input_files:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
                    outfile.write(f"\n\n--- Content from {os.path.basename(file_path)} ---\n\n")
                    outfile.write(infile.read())
            else:
                print(f"⚠️ File not found: {file_path}")

    print(f"✅ All files combined into: {output_file}")

# Run the function
combine_files(file_paths, output_file)
