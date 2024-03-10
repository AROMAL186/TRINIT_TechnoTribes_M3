import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torchvision.transforms as transforms
import random
from nltk.translate.bleu_score import corpus_bleu
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the data
df_train = pd.read_csv('clean/train_clean.csv')
df_val = pd.read_csv('clean/val_clean.csv')
df_test = pd.read_csv('clean/test_clean.csv')

# Define the tokenizer and model
processor = AutoTokenizer.from_pretrained("microsoft/git-base")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

# Define transforms for preprocessing images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a fixed size
    transforms.ToTensor(),           # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize pixel values
])

# Define the dataset class
class ImageCaptioningDataset(Dataset):
    def __init__(self, df, processor, transform):
        self.file_paths = df['filepath'].tolist()
        self.captions = df['captions_list'].apply(eval).tolist()
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fpath = os.path.join('archive', self.file_paths[idx])
        image = Image.open(fpath).convert('RGB')
        image = self.transform(image)

        caption = random.choice(self.captions[idx])
        tokenized_caption = self.processor.encode_plus(
            caption,
            padding="max_length",
            max_length=50,
            return_tensors="pt",
            truncation=True
        )

        return {
            'image': image,
            'caption_input_ids': tokenized_caption['input_ids'].squeeze(),
            'caption_attention_mask': tokenized_caption['attention_mask'].squeeze()
        }

# Custom collate function
def custom_collate(batch):
    images = torch.stack([item['image'] for item in batch])
    caption_input_ids = torch.stack([item['caption_input_ids'] for item in batch])
    caption_attention_mask = torch.stack([item['caption_attention_mask'] for item in batch])

    return {
        'images': images,
        'caption_input_ids': caption_input_ids,
        'caption_attention_mask': caption_attention_mask
    }

# Create datasets and dataloaders
train_dataset = ImageCaptioningDataset(df_train, processor, transform)
val_dataset = ImageCaptioningDataset(df_val, processor, transform)
test_dataset = ImageCaptioningDataset(df_test, processor, transform)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=custom_collate)
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=8, collate_fn=custom_collate)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=4, collate_fn=custom_collate)

# Define optimizer and move model to device
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values=image, max_length=50)
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)

    return render_template('result.html', caption=caption)

if __name__ == '__main__':
    app.run(debug=True)
