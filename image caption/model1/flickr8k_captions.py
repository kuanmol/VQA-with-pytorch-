import os
import re
from collections import defaultdict

def load_flickr8k_captions(caption_file):
    id_to_captions = defaultdict(list)
    with open(caption_file, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            image_caption_id, caption = line.split('\t')
            image_id = image_caption_id.split('#')[0]  # e.g. '1000268201_693b08cb0e.jpg'
            caption = clean_caption(caption)
            id_to_captions[image_id].append(caption)
    return id_to_captions

def clean_caption(caption):
    caption = caption.lower()
    caption = re.sub(r"[^a-z ]+", "", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption
