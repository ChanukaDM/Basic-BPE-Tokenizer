def get_stats(ids):
    counts = {}
    for pair in zip(ids,ids[1:]):
        counts[pair] = counts.get(pair,0) + 1
    return counts

def merge(ids,pair,idx):
    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class BasicTokenizer:
    def __init__(self,text,vocab_size):
        super().__init__()
        self.text = text
        self.vocab_size = vocab_size
        self.merges = {}
        self.num_merges = vocab_size - 256
        self.ids = list(map(int,text.encode("utf-8")))


    def __call__(self):

        for i in range(self.num_merges):
            stats = get_stats(self.ids)
            pair = max(stats, key = stats.get)
            new_idx = 256 + i
            ids = merge(self.ids,pair,new_idx)
            self.merges[pair] = new_idx
            self.merges.append(new_idx)
        return self.merges

    def decode(self,ids):

        vocab = {idx: bytes([idx]) for idx in range (256)}
        for (p0,p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]  #concat

        tokens = b"".join(vocab[idx] for idx in ids)  # adding a b initially is a way of byte concatanation
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    def encode(self,raw_text):
        tokens = list(raw_text.encode("utf-8"))

        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p,float('inf')))
            if pair not in self.merges:
                break # Nothing else can be merged
            idx = self.merges[pair]
            tokens = merge(tokens,pair,idx)
        return tokens
    


with open(r'C:\Users\User\OneDrive\Desktop\DL_Project\Tokenizer\train_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()


tokenizer = BasicTokenizer(text,500)

test_encode = tokenizer.encode("Hello World")
test_encode2 = tokenizer.encode("hello world") # Check lowercase sensitivity

test_decode = tokenizer.decode(test_encode2)


print(test_encode)
print(test_decode)
print(test_encode == test_encode2)




