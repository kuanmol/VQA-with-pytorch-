import random

def split_dataset(image_ids, train_ratio=0.8, val_ratio=0.1):
    random.shuffle(image_ids)
    n_total = len(image_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:n_train + n_val]
    test_ids = image_ids[n_train + n_val:]
    return train_ids, val_ids, test_ids