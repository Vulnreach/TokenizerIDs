import re 

with open ("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("total of character :", len(raw_text))
print(raw_text[:99])

 # ---------------------------------------------------------- //

text = "Hello, world. Is this-- a test?"
result2 = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result2 = [item.strip() for item in result2 if item.strip()]
print(result2)

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:100000])



# ------- CREATE TOKEN IDs based on TOKEN -------------# 

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size)

vocab = {token:integer for integer,token in enumerate (all_words)}

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


# 👇 Maintenant tu peux instancier
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
