import pandas as pd
import ast
import os
import re

# Read the train.csv file
df = pd.read_csv('archive/valid.csv')

# Path to the images and captions folders
images_folder = 'images'
captions_folder = 'valid_captions'

# Create the images folder if it doesn't exist
if not os.path.exists(images_folder):
    os.makedirs(images_folder)

# Create the captions folder if it doesn't exist
if not os.path.exists(captions_folder):
    os.makedirs(captions_folder)

# Process the datasets
for i, row in df.iterrows():
    try:
        # Get the image bytes and filename
        # Get the captions and clean them
        captions = row['captions']
        captions = re.sub(r'\[.*?\]', '', captions)  # Remove brackets and inner content
        captions = re.sub(r'[\r\n]+', '\n', captions)  # Normalize line endings
        captions = [caption.strip() for caption in captions.split('\n') if caption.strip()]

        # Save the captions to a text file
        caption_filename = f"valid_cap_{i:05d}.txt"  # Use 5 digits with leading zeros
        caption_path = os.path.join(captions_folder, caption_filename)
        with open(caption_path, 'w') as f:
            for caption in captions:
                f.write("%s\n" % caption)
    except Exception as e:
        print(f"Error processing row {i}: {e}")
