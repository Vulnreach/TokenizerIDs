import re 

# --- Lire le texte brut ---
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("total of character :", len(raw_text))
print(raw_text[:99])

# ---------------------------------------------------------- #

# Exemple simple
text = "Hello, world. Is this-- a test?"
result2 = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result2 = [item.strip() for item in result2 if item.strip()]
print(result2)

# Tokenisation du texte complet
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:1000])

# --- Tokens spéciaux ---
SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
}

# ------- Créer le vocabulaire -------------
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print("Taille vocab :", vocab_size)

vocab = {token: i + len(SPECIAL_TOKENS) for i, token in enumerate(all_words)}
vocab = {**SPECIAL_TOKENS, **vocab}  # merge

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

# --- Tokenizer V2 ---
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int.get(s, self.str_to_int["<UNK>"]) for s in preprocessed]
        return [self.str_to_int["<BOS>"]] + ids + [self.str_to_int["<EOS>"]]
        
    def decode(self, ids):
        tokens = [self.int_to_str[i] for i in ids if i not in {0, 2, 3}]  
        text = " ".join(tokens)
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

    @staticmethod
    def pad_sequences(sequences, max_len, pad_value=0):
        return [
            seq + [pad_value] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]
            for seq in sequences
        ]


# --- Test ---
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
