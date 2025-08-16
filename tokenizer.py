import re
import json

# 1) Lire le corpus
with open("dictionnaire_academie_francaise_5eme_edition.txt", "r", encoding="utf-8") as f_in:
    raw_text = f_in.read()

print("total of character :", len(raw_text))
print(raw_text[:99])

# 2) Tokeniser
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# 3) Tokens spéciaux + vocab
SPECIAL_TOKENS = {"<PAD>":0, "<UNK>":1, "<BOS>":2, "<EOS>":3}
all_words = sorted(set(preprocessed))
vocab = {**SPECIAL_TOKENS,
         **{tok: i + len(SPECIAL_TOKENS) for i, tok in enumerate(all_words)}}

print("Taille vocab :", len(vocab))

# 4) Sauvegarder le vocab (APRES l’avoir construit)
with open("vocab.json", "w", encoding="utf-8") as f_out:
    json.dump(vocab, f_out, ensure_ascii=False, indent=2)

# 5) Recharger si besoin
with open("vocab.json", "r", encoding="utf-8") as f_in:
    vocab = json.load(f_in)

# 6) Tokenizer
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        toks = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        toks = [t.strip() for t in toks if t.strip()]
        ids = [self.str_to_int.get(t, self.str_to_int["<UNK>"]) for t in toks]
        return [self.str_to_int["<BOS>"]] + ids + [self.str_to_int["<EOS>"]]

    def decode(self, ids):
        tokens = [self.int_to_str[i] for i in ids if self.int_to_str.get(i) not in {"<PAD>","<BOS>","<EOS>"}]
        text = " ".join(tokens)
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text)

    @staticmethod
    def pad_sequences(sequences, max_len, pad_value=0):
        return [seq + [pad_value]*(max_len-len(seq)) if len(seq) < max_len else seq[:max_len] for seq in sequences]

# 7) Test
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
