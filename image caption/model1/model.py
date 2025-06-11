import torch
import torch.nn as nn
import torchvision.models as models
import random

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.embed_size = embed_size
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.conv_proj = nn.Linear(2048, embed_size)
        self.conv_bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.feat_proj = nn.Linear(2048, embed_size)
        self.feat_bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.dropout = nn.Dropout(0.4)  # Increased for regularization

    def forward(self, x):
        if x.dim() == 2:
            feats = self.feat_proj(x)
            feats = self.feat_bn(feats)
            feats = self.dropout(feats)
            return feats.unsqueeze(1)
        feats = self.resnet(x)
        feats = self.adaptive_pool(feats)
        B, C, H, W = feats.size()
        feats = feats.view(B, C, -1).permute(0, 2, 1)
        feats = self.conv_proj(feats)
        feats = feats.view(-1, self.embed_size)
        feats = self.conv_bn(feats)
        feats = self.dropout(feats)
        feats = feats.view(B, -1, self.embed_size)
        return feats

class Attention(nn.Module):
    def __init__(self, encoder_dim, hidden_dim, attention_dim, num_heads=4):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.enc_att = nn.Linear(encoder_dim, attention_dim)
        self.dec_att = nn.Linear(hidden_dim, attention_dim)
        self.query = nn.Linear(attention_dim, attention_dim)
        self.key = nn.Linear(attention_dim, attention_dim)
        self.value = nn.Linear(attention_dim, attention_dim)
        self.fc = nn.Linear(attention_dim, encoder_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encoder_out, hidden_state):
        batch_size = encoder_out.size(0)
        seq_len = encoder_out.size(1)
        enc_out = self.enc_att(encoder_out)
        dec_out = self.dec_att(hidden_state).unsqueeze(1)
        query = self.query(dec_out).view(batch_size, 1, self.num_heads, self.attention_dim // self.num_heads).permute(0, 2, 1, 3)
        key = self.key(enc_out).view(batch_size, seq_len, self.num_heads, self.attention_dim // self.num_heads).permute(0, 2, 1, 3)
        value = self.value(enc_out).view(batch_size, seq_len, self.num_heads, self.attention_dim // self.num_heads).permute(0, 2, 1, 3)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.attention_dim // self.num_heads) ** 0.5
        attn_weights = self.softmax(scores)
        context = torch.matmul(attn_weights, value).permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, 1, self.attention_dim)
        context = self.fc(context.squeeze(1))
        return context, attn_weights

class DecoderWithAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim=256, encoder_dim=256, num_layers=2):
        super(DecoderWithAttention, self).__init__()
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(embed_size + encoder_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        self.num_layers = num_layers

    def init_hidden_state(self, encoder_out):
        mean_enc = encoder_out.mean(dim=1)
        h = self.init_h(mean_enc).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = self.init_c(mean_enc).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h, c

    def forward(self, encoder_out, captions, teacher_forcing_ratio=0.5):
        batch_size = encoder_out.size(0)
        embeddings = self.embedding(captions)
        h, c = self.init_hidden_state(encoder_out)
        outputs = []
        alphas = []
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        input_emb = embeddings[:, 0, :].unsqueeze(1)
        for t in range(captions.size(1) - 1):
            attn_weighted, alpha = self.attention(encoder_out, h[-1])
            alphas.append(alpha)
            lstm_input = torch.cat([input_emb, attn_weighted.unsqueeze(1)], dim=2)
            output, (h, c) = self.lstm(lstm_input, (h, c))
            output = self.fc(self.dropout(output.squeeze(1)))
            outputs.append(output)
            if use_teacher_forcing:
                input_emb = embeddings[:, t + 1, :].unsqueeze(1)
            else:
                _, topi = output.topk(1)
                input_emb = self.embedding(topi.squeeze()).unsqueeze(1)
        outputs = torch.stack(outputs, dim=1)
        alphas = torch.stack(alphas, dim=1)
        return outputs, alphas

    def sample(self, encoder_out, max_len=20):
        batch_size = encoder_out.size(0)
        h, c = self.init_hidden_state(encoder_out)
        sampled_ids = []
        inputs = self.embedding(
            torch.full((batch_size,), self.embedding.weight.size(0) - 1, dtype=torch.long, device=encoder_out.device)
        ).unsqueeze(1)
        for _ in range(max_len):
            attn_weighted, alpha = self.attention(encoder_out, h[-1])
            lstm_input = torch.cat([inputs, attn_weighted.unsqueeze(1)], dim=2)
            output, (h, c) = self.lstm(lstm_input, (h, c))
            output = self.fc(output.squeeze(1))
            _, predicted = output.max(1)
            sampled_ids.append(predicted)
            inputs = self.embedding(predicted).unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids