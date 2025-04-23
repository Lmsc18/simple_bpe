from utils import getstat,merge

class Tokenizer():
    def __init__(self):
        super().__init__()
    
    def train(self,text_file:str,vocab_size:str):
        assert vocab_size>=256
        num_merges=vocab_size-256
        with open(f"{text_file}",'r',encoding='utf-8') as f:
            text=f.read()
        text_bytes=text.encode("utf-8")
        ids=list(text_bytes)
        merges={}
        vocab={idx:bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats=getstat(ids)
            pair=max(stats,key=stats.get)
            idx=256+i
            ids=merge(ids=ids,pair=pair,idx=idx)
            merges[pair]=idx
            vocab[idx]=vocab[pair[0]]+vocab[pair[1]]
        self.merges=merges
        self.vocab=vocab

    def encode(self,text:str):
        ids=list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats=getstat(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx=self.merges[pair]
            ids=merge(ids,pair,idx)
        return ids
    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    def save(self, file_prefix):
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            f.write("minbpe v1\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        merges = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            version = f.readline().strip()
            assert version == "minbpe v1"
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges