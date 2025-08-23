import re
import json
from collections import defaultdict, Counter

# 1) Lire le corpus
with open("the-verdict.txt", "r", encoding="utf-8") as f_in:
    raw_text = f_in.read()
print("total of character :", len(raw_text))
print(raw_text[:99])


# PRE tokenizer 
class ImprovedPreTokenizer:
    def __init__(self):
        # pattern pour les différent type de token 
     self.patterns = {
            'whitespace': r'\s+',
            'numbers': r'\d+\.?\d*',
            'urls': r'https?://[^\s]+',
            'emails': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'mentions': r'@\w+',
            'hashtags': r'#\w+',
            'contractions': r"n't|'re|'ve|'ll|'d|'m|'s",
            'punctuation': r'[.,!?;:()\[\]{}"\'`~]',
            'special_chars': r'[-_+=<>/\\|*&^%$#@!]',
            'cve': r'CVE-\d{4}-\d{4,7}',  # Ajout pour vuln
            'ip': r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # Ajout pour logs réseau
            'base64': r'[A-Za-z0-9+/=]{20,}',  # Ajout pour payloads
            'words': r'\w+',
        }

# 2) Tokeniser
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# 3) Tokens spéciaux + vocab
SPECIAL_TOKENS = {"<PAD>":0, "<UNK>":1, "<BOS>":2, "<EOS>":3}
all_words = sorted(set(preprocessed))
vocab = {**SPECIAL_TOKENS,
         **{tok: i + len(SPECIAL_TOKENS) for i, tok in enumerate(all_words)}}
print("Taille vocab :", len(vocab))

# 4) Sauvegarder le vocab (APRES l'avoir construit)
with open("vocab.json", "w", encoding="utf-8") as f_out:
    json.dump(vocab, f_out, ensure_ascii=False, indent=2)

# 5) Recharger si besoin
with open("vocab.json", "r", encoding="utf-8") as f_in:
    vocab = json.load(f_in)

# 6) Tokenizer simple
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

    @staticmethod
    def analyse_tokenizer(tokenizer, test_text):
        for text in test_text:
            ids = tokenizer.encode(text)
            compression_ratio = len(text) / len(ids)
            print(f"Texte: {text[:50]}...")
            print(f"Tokens: {len(ids)}, Ratio: {compression_ratio:.2f}\n")

# 7) BPE Tokenizer
class ByteLevelBPETokenizer:
    def __init__(self):
        self.encoder = {}  # str -> int
        self.decoder = {}  # int -> str
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = {}  # paires -> rang de merge
        self.special_tokens = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        
    def _bytes_to_unicode(self):
        """Mapping GPT-2 style : 256 bytes -> caractères Unicode printables"""
        bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8+n)
                n += 1
        return dict(zip(bs, [chr(n) for n in cs]))
        
    def _get_pairs(self, word):
        """Extrait toutes les paires adjacentes dans un mot"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
        
    def _bpe(self, token):
        """Applique l'algorithme BPE sur un token"""
        if len(token) < 2:
            return token
            
        word = tuple(token)
        pairs = self._get_pairs(word)
        
        if not pairs:
            return token
            
        while True:
            # Trouve la paire avec le rang le plus bas (merge en premier)
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
                
            # Applique le merge
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                    
                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
                    
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)
            
        return ' '.join(word)
        
    def train_bpe(self, text, vocab_size=50000, min_frequency=2):
        """Entraîne le BPE sur un corpus"""
        # 1. Pre-tokenization et conversion en bytes
        words = re.findall(r'\S+', text)
        word_freqs = Counter(words)
        
        # 🔍 DIAGNOSTIC 1
        print(f" Mots uniques dans le corpus: {len(word_freqs)}")
        print(f"Total mots: {sum(word_freqs.values())}")
        
        # 2. Conversion en représentation byte-level
        vocab = set()
        for word in word_freqs:
            word_bytes = word.encode('utf-8')
            word_unicode = ''.join(self.byte_encoder[b] for b in word_bytes)
            word_unicode = word_unicode + '</w>'
            vocab.update(word_unicode.split())
            
        # 3. Vocabulaire initial : tous les caractères uniques
        vocab = list(vocab)
        
        # 🔍 DIAGNOSTIC 2
        print(f" Vocabulaire initial (caractères): {len(vocab)}")
        
        splits = {}
        for word in word_freqs:
            word_bytes = word.encode('utf-8')
            word_unicode = ''.join(self.byte_encoder[b] for b in word_bytes) + '</w>'
            splits[word] = word_unicode.split()
            
        # 4. Itérations BPE
        num_merges = vocab_size - len(vocab) - len(self.special_tokens)
        
        # 🔍 DIAGNOSTIC 3
        print(f" Target: {vocab_size} tokens")
        print(f"Merges prévus: {num_merges}")
        
        for i in range(num_merges):
            # 🔍 DIAGNOSTIC 4 (progress)
            if i % 1000 == 0 and i > 0:
                print(f"  Merge {i}/{num_merges}")
                
            # Compter toutes les paires
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                symbols = splits[word]
                for j in range(len(symbols) - 1):
                    pairs[(symbols[j], symbols[j + 1])] += freq
                    
            if not pairs:
                print(f" Arrêt: plus de paires disponibles à {i} merges")
                break
                
            # Trouver la paire la plus fréquente
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < min_frequency:
                print(f" Arrêt: fréquence trop basse ({pairs[best_pair]}) à {i} merges")
                break
                
            # Enregistrer le merge
            self.bpe_ranks[best_pair] = i
            first, second = best_pair
            new_splits = {}
            for word in splits:
                symbols = splits[word]
                new_symbols = []
                j = 0  # CHANGÉ: utiliser j au lieu de i pour éviter conflit
                while j < len(symbols):
                    if j < len(symbols) - 1 and symbols[j] == first and symbols[j + 1] == second:
                        new_symbols.append(first + second)
                        j += 2
                    else:
                        new_symbols.append(symbols[j])
                        j += 1
                new_splits[word] = new_symbols
            splits = new_splits
            vocab.append(first + second)
        
        # 🔍 DIAGNOSTIC 5 - Final
        print(f" Vocabulaire final: {len(vocab) + len(self.special_tokens)} tokens")
        
        # 5. Construire l'encodeur final
        self.encoder = {**self.special_tokens}
        for i, token in enumerate(vocab):
            self.encoder[token] = len(self.special_tokens) + i
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        print(f"Vocabulaire BPE entraîné : {len(self.encoder)} tokens")
        return self.encoder
        
    def encode(self, text):
        """Encode un texte en tokens"""
        if not self.encoder:
            raise ValueError("Le tokenizer n'est pas encore entraîné ! Utilisez train_bpe() d'abord.")
            
        tokens = []
        for word in re.findall(r'\S+', text):
            # Convertir en byte-level
            word_bytes = word.encode('utf-8')
            word_unicode = ''.join(self.byte_encoder[b] for b in word_bytes) + '</w>'
            
            # Appliquer BPE
            bpe_tokens = self._bpe(word_unicode).split()
            
            # Convertir en IDs
            for token in bpe_tokens:
                tokens.append(self.encoder.get(token, self.encoder['<UNK>']))
                
        return [self.encoder['<BOS>']] + tokens + [self.encoder['<EOS>']]
        
    def decode(self, ids):
        """Décode une liste d'IDs en texte"""
        tokens = []
        for id in ids:
            if id not in [self.encoder['<PAD>'], self.encoder['<BOS>'], self.encoder['<EOS>']]:
                token = self.decoder.get(id, '<UNK>')
                tokens.append(token)
                
        # Reconstituer le texte
        text = ''.join(tokens).replace('</w>', ' ')
        
        # Convertir depuis byte-level vers UTF-8
        try:
            # Convertir depuis les caractères Unicode vers bytes
            byte_sequence = []
            for char in text:
                if char in self.byte_decoder:
                    byte_sequence.append(self.byte_decoder[char])
                    
            # Décoder en UTF-8
            text = bytes(byte_sequence).decode('utf-8', errors='ignore')
        except:
            pass
            
        return text.strip()

# 8) Test
print("\n=== Test SimpleTokenizer ===")
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print("IDs:", ids)
print("Décodé:", tokenizer.decode(ids))

# Test avec du texte cyber
cyber_samples = [
    "CVE-2021-44228 affects Apache Log4j versions 2.0-beta9",
    "192.168.1.100:8080 returned HTTP 200 with SQLi payload",
    "Base64 payload: dGVzdCBwYXlsb2FkCg== executed successfully"
]
SimpleTokenizerV1.analyse_tokenizer(tokenizer, cyber_samples)

print("\n=== Test BPE Tokenizer ===")
bpe_tokenizer = ByteLevelBPETokenizer()
# Entraîner sur un échantillon du texte
sample_text = raw_text[:50000]  # Premier 50k caractères pour test
bpe_tokenizer.train_bpe(sample_text, vocab_size=10000)

# Test d'encodage/décodage
test_text = "Bonjour le monde! Comment allez-vous?"
bpe_ids = bpe_tokenizer.encode(test_text)
print("BPE IDs:", bpe_ids)
print("BPE Décodé:", bpe_tokenizer.decode(bpe_ids))
