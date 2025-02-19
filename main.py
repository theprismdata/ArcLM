import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import re
from typing import List, Tuple, Dict
from konlpy.tag import Mecab  # 한글 형태소 분석기

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Multi-head attention layer with scaled dot-product attention
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)

        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Initialize parameters with Xavier uniform
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create padding mask for attention"""
        return src == 0  # PAD token index is 0

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        if src_mask is None:
            src_mask = self.create_mask(x)

        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)

        # Transformer layers
        x = self.transformer(x, src_key_padding_mask=src_mask)

        # Output projection
        return self.fc_out(x)

class BusinessProposalTokenizer:
    def __init__(self, vocab_size=2000):
        self.vocab_size = vocab_size
        self.token2idx = {}
        self.idx2token = {}
        self.mecab = Mecab()

    def fit(self, texts: List[str]):
        # 형태소 분석을 통한 토큰화
        word_freq = {}
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1

        # 빈도수 기준으로 상위 vocab_size개의 토큰 선정
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for idx, (word, freq) in enumerate(sorted_words[:self.vocab_size - 4]):
            self.token2idx[word] = idx + 4
            self.idx2token[idx + 4] = word

        # 특수 토큰 추가
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        for idx, token in enumerate(special_tokens):
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def tokenize(self, text: str) -> List[str]:
        """텍스트를 형태소 단위로 분리"""
        text = self.preprocess_text(text)
        # 형태소 분석 결과를 토큰으로 변환
        morphs = self.mecab.morphs(text)
        return morphs

    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리 함수"""
        # 불필요한 공백 제거
        text = ' '.join(text.split())
        # 특수문자 처리 (마침표, 쉼표, 괄호 등은 유지)
        text = re.sub(r'[^\w\s\.,\(\)\[\]\"\';:]', ' ', text)
        return text.strip()

    def encode(self, text: str) -> List[int]:
        """텍스트를 토큰 인덱스로 변환"""
        tokens = self.tokenize(text)
        # BOS, EOS 토큰 추가
        return [self.token2idx['<BOS>']] + [self.token2idx.get(token, self.token2idx['<UNK>']) for token in tokens] + [
            self.token2idx['<EOS>']]

    def decode(self, indices: List[int]) -> str:
        """토큰 인덱스를 텍스트로 변환"""
        tokens = [self.idx2token.get(idx, '<UNK>') for idx in indices]
        # 특수 토큰 제거
        tokens = [token for token in tokens if token not in ['<PAD>', '<BOS>', '<EOS>', '<UNK>']]
        return ' '.join(tokens)


class ProposalDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: BusinessProposalTokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        for text in texts:
            # 문장을 토큰 인덱스로 변환
            tokens = self.tokenizer.encode(text)
            if len(tokens) > max_len:
                tokens = tokens[:max_len - 1] + [self.tokenizer.token2idx['<EOS>']]
            else:
                tokens = tokens + [self.tokenizer.token2idx['<PAD>']] * (max_len - len(tokens))
            self.data.append(tokens)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.tensor(self.data[idx])
        return tokens[:-1], tokens[1:]


def load_and_preprocess_data(file_path: str) -> List[str]:
    """텍스트 파일을 읽고 전처리하는 함수"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 문장 단위로 분리 (마침표, 물음표, 느낌표 기준)
    sentences = re.split(r'[.!?]\s+', text)
    # 빈 문장 제거 및 최소 길이 필터링
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences


def train_model(model: nn.Module, train_loader: DataLoader, epochs: int = 10, device: str = 'cuda'):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # <PAD> 토큰 무시

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')


def generate_proposal(model: nn.Module, tokenizer: BusinessProposalTokenizer,
                      start_text: str, max_len: int = 100, temperature: float = 0.7) -> str:
    model.eval()
    tokens = tokenizer.encode(start_text)

    with torch.no_grad():
        for _ in range(max_len):
            src = torch.tensor(tokens).unsqueeze(0).to(next(model.parameters()).device)
            output = model(src)
            next_token_logits = output[0, -1, :] / temperature
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, 1).item()

            if next_token == tokenizer.token2idx['<EOS>']:
                break

            tokens.append(next_token)

    return tokenizer.decode(tokens)


def main():
    # 설정
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    VOCAB_SIZE = 8000  # 한글 텍스트를 위해 확장
    BATCH_SIZE = 32
    MAX_LEN = 256
    EPOCHS = 100

    print(f"Using device: {DEVICE}")

    # 1. 데이터 로드 및 전처리
    print("Loading and preprocessing data...")
    texts = load_and_preprocess_data('2014-05-정책효과성 증대를 위한 집행과학에 관한 연구.pdf.txt')
    print(f"Loaded {len(texts)} sentences")

    # 2. 토크나이저 학습
    print("Training tokenizer...")
    tokenizer = BusinessProposalTokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.fit(texts)
    print(f"Vocabulary size: {len(tokenizer.token2idx)}")

    # 3. 데이터셋 및 데이터로더 생성
    print("Creating dataset and dataloader...")
    dataset = ProposalDataset(texts, tokenizer, max_len=MAX_LEN)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. 모델 초기화
    print("Initializing model...")
    model = TinyTransformer(vocab_size=VOCAB_SIZE).to(DEVICE)

    # 5. 모델 학습
    print("Training model...")
    train_model(model, train_loader, epochs=EPOCHS, device=DEVICE)

    # 6. 모델 저장
    print("Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'vocab_size': VOCAB_SIZE,
        'model_config': {
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2
        }
    }, 'business_proposal_model.pth')

    # 7. 테스트 생성
    print("\nGenerating sample text...")
    sample_prompts = [
        "사업 계획서의 목적은",
        "본 사업의 기대효과는",
        "시장 분석 결과",
    ]

    for prompt in sample_prompts:
        generated_text = generate_proposal(model, tokenizer, prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")


def test():
    checkpoint = torch.load('business_proposal_model.pth')
    model = TinyTransformer(vocab_size=checkpoint['vocab_size'],
                            **checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    tokenizer = checkpoint['tokenizer']

    text = generate_proposal(model, tokenizer, "사업 계획서의 목적은")
    print(text)
if __name__ == "__main__":
    main()
    test()
