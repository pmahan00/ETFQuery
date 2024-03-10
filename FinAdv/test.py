import csv

csv_path = "/home/tonoi/FinAdv/test/data.csv"

def process_comments(comments):
    # Strip leading and trailing whitespace from each comment
    # and join them into a single string with a space character between each comment
    # This will create a more continuous block of text without extra spaces or blank lines
    return " ".join(comment.strip() for comment in comments)

with open(csv_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Construct the file name using date and popularity
        file_name = f"date{row['date']}_popularity{row['popularity']}.md"

        with open(file_name, 'w') as mdfile:
            # Title section
            mdfile.write(f"# {row['title']}\n\n")

            # Description section
            mdfile.write(f"## Description\n\n{row['description']}\n\n")

            # Reference (URL) section
            mdfile.write(f"## Reference\n\n[Reference]({row['reference']})\n\n")

            # Popularity section
            mdfile.write(f"## Popularity\n\n{row['popularity']}\n\n")

            # Date section
            mdfile.write(f"## Date\n\n{row['date']}\n\n")

            # Comments section
            comments = row['comments'].split(',') # Assuming comments are comma-separated
            comment_block = process_comments(comments)
            # Make the comments bold under a different header
            mdfile.write(f"### Comments\n\n**{comment_block}**\n\n")
