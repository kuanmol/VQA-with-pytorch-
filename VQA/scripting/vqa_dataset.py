import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from collections import Counter

class SimpleVQADataset(Dataset):
    """
    A minimal VQA dataset loader: single most common answer per question, basic vocab.
    """
    def __init__(self, questions_json, annotations_json, image_dir, max_q_len=20):
        # Load JSON data
        with open(questions_json, 'r') as fq:
            questions = json.load(fq)['questions']
        with open(annotations_json, 'r') as fa:
            anns = json.load(fa)['annotations']

        self.questions = questions
        # Map question_id -> list of answers
        self.anns = {a['question_id']: [ans['answer'].lower() for ans in a['answers']] for a in anns}
        self.image_dir = image_dir
        self.max_q_len = max_q_len

        # Build simple word vocab
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        idx = 2
        for q in self.questions:
            for w in q['question'].lower().split():
                if w.isalpha() and w not in self.word2idx:
                    self.word2idx[w] = idx
                    idx += 1

        # Build answer vocab (top 500 answers)
        all_answers = [ans for answers in self.anns.values() for ans in answers]
        common = [a for a, _ in Counter(all_answers).most_common(500)]
        # Reserve index 0 as <unk answer>
        self.ans2idx = {a: i+1 for i, a in enumerate(common)}
        self.idx2ans = ['<unk>'] + common

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, i):
        q = self.questions[i]
        qid, img_id = q['question_id'], q['image_id']

        # Load and preprocess image
        fname = f"COCO_train2014_{img_id:012d}.jpg"
        img = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
        img = self.transform(img)

        # Encode question
        tokens = q['question'].lower().split()
        ids = [self.word2idx.get(t, 1) for t in tokens][:self.max_q_len]
        ids += [0] * (self.max_q_len - len(ids))
        q_tensor = torch.tensor(ids, dtype=torch.long)

        # Single-label target: most common answer
        answers = self.anns[qid]
        label_str = Counter(answers).most_common(1)[0][0]
        # Map unseen answers to 0 (<unk>)
        label_idx = self.ans2idx.get(label_str, 0)
        y = torch.tensor(label_idx, dtype=torch.long)
        return img, q_tensor, y