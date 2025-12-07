import torch
import os
import kagglehub
from clip import CLIP, CLIPLoss
from data_loader import get_data_loader
from vision_transformer import VisionTransformer
from text_transformer import TextTransformer

# Training configuration
original_path = kagglehub.dataset_download("adityajn105/flickr30k")
images_dir = os.path.join(os.path.dirname(__file__), '..', 'test_images')
captions_file = os.path.join(os.path.dirname(__file__), '..', 'test_captions.txt')
batch_size = 32
num_workers = 2
num_epochs = 10
learning_rate = 1e-4

train_loader = get_data_loader(images_dir, captions_file, batch_size, num_workers, max_samples=10)

# Model
model = CLIP()
loss_fn = CLIPLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def print_first_10_pairs():
    print("First 10 image-caption pairs:")
    count = 0
    for images, texts in train_loader:
        for i in range(images.shape[0]):
            if count >= 10:
                return
            img_info = f"Image tensor shape: {images[i].shape}"
            txt_info = f"Tokenized caption: {texts[i].tolist()}"
            print(f"Pair {count+1}: {img_info}, {txt_info}")
            count += 1

def train():
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, captions in train_loader:
            optimizer.zero_grad()
            logits, image_features, text_features = model(images, captions)
            loss = loss_fn(logits)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    print_first_10_pairs()
    train()