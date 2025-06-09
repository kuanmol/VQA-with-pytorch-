import os
import string
import pickle
import re

CAPTIONS_FILE = r"D:\Projects\asfdgfhjghk\image caption\dataset\Flickr8k_text\Flickr8k.token.txt"
OUTPUT_FILE = r"D:\Projects\asfdgfhjghk\image caption\data\captions.pkl"

def clean_caption(caption):
    caption = caption.lower()
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    caption = re.sub(r"[^a-z ]+", "", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    return f"<start> {caption} <end>"

def load_captions(captions_file):
    captions_dict = {}
    try:
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ', 1)
                if len(parts) < 2:
                    continue
                raw_id, caption = parts
                image_name = raw_id.split('#')[0]
                caption = clean_caption(caption)
                captions_dict.setdefault(image_name, []).append(caption)
    except FileNotFoundError:
        print(f"Error: Captions file not found at {captions_file}")
        raise
    return captions_dict

def save_captions(captions, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(captions, f)
    print(f"✅ Captions saved to {output_file}")

if __name__ == "__main__":
    captions = load_captions(CAPTIONS_FILE)
    save_captions(captions, OUTPUT_FILE)