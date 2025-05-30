import torch
from torch.utils.data import Dataset
import pickle
import re

class Flickr8kDataset(Dataset):
    def __init__(self, image_ids, image_features_path, captions_path, vocab=None, max_len=20):
        self.image_ids = image_ids
        self.image_features = pickle.load(open(image_features_path, 'rb'))
        self.captions = pickle.load(open(captions_path, 'rb'))
        self.max_len = max_len

        # Build or use given vocab
        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab

        # Prepare flattened data: list of (img_name, token_ids)
        self.data = []
        for img_id in self.image_ids:
            base_id = re.match(r'^(.+?\.jpg)', img_id).group(1)
            for caption in self.captions.get(base_id, []):
                token_ids = self.encode_caption(caption)
                self.data.append((base_id, token_ids))

    def build_vocab(self):
        words = set(['<start>', '<end>', '<unk>'])
        for caps in self.captions.values():
            for cap in caps:
                cap = cap.replace('<start>', '').replace('<end>', '').strip()
                words.update(cap.split())
        words = sorted(words)
        vocab = {word: idx + 1 for idx, word in enumerate(words)}
        vocab['<pad>'] = 0
        return vocab

    def encode_caption(self, caption):
        tokens = caption.split()
        ids = [self.vocab.get(tok, self.vocab['<unk>']) for tok in tokens]
        if len(ids) < self.max_len:
            ids += [self.vocab['<pad>']] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption_ids = self.data[idx]
        feature = self.image_features[img_name]
        return feature, caption_ids

def collate_fn(batch):
    features, captions = zip(*batch)
    features = torch.stack(features, 0)
    lengths = [len(c) for c in captions]
    max_len = max(lengths)
    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        end = lengths[i]
        padded[i, :end] = cap
    return features, padded, lengths

if __name__ == "__main__":
    sample_ids = ['1000268201_693b08cb0e.jpg', '1001773457_577c3a7d70.jpg']
    dataset = Flickr8kDataset(
        image_ids=sample_ids,
        image_features_path="../data/image_features.pkl",
        captions_path="../data/captions.pkl"
    )
    print(f"Dataset size: {len(dataset)}")
    feat, cap = dataset[0]
    print(f"Feature shape: {feat.shape}")
    print(f"Caption tokens: {cap}")