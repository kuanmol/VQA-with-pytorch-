import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pickle
from tqdm import tqdm

IMAGES_FOLDER = r"D:\Projects\asfdgfhjghk\image caption\dataset\Flicker8k_Dataset"
OUTPUT_FILE = r"D:\Projects\asfdgfhjghk\image caption\data\image_features.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
modules = list(resnet.children())[:-1]
resnet = torch.nn.Sequential(*modules)
resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_feature(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = resnet(image).squeeze()
    return feature.cpu()

def extract_all_features(image_folder):
    features = {}
    for img_name in tqdm(os.listdir(image_folder)):
        if not img_name.lower().endswith(".jpg"):
            continue
        img_path = os.path.join(image_folder, img_name)
        features[img_name] = extract_feature(img_path)
    return features

def save_features(features, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(features, f)
    print(f"✅ Features saved to {output_file}")

if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)
    features = extract_all_features(IMAGES_FOLDER)
    save_features(features, OUTPUT_FILE)