import re
import json
import pickle

def read_data(data_file):
    with open(data_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        enc_data, dec_data = [], []
        for line in lines:
            if line == "":
                continue
            enc, dec = line.split("\t")
            enc_data.append(re.findall(r"[\w']+", enc))
            dec_data.append(["<BOS>"] + re.findall(r"[\u4e00-\u9fff]", dec) + ["<EOS>"])
    assert len(enc_data) == len(dec_data)
    return enc_data, dec_data


class Vocabulary:
    
    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def from_documents(cls, documents):
        tokens = set() 
        for cmt in documents:
            tokens.update(list(cmt))
        tokens.discard("<BOS>") # 在dec_data中已经存在"<BOS>", "<EOS>",需要先删除，否则会分别有两个"<BOS>", "<EOS>"
        tokens.discard("<EOS>")
        # set是无序的，可以在list之后做排序,保证每次构建词典顺序一致
        tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"] + sorted(list(tokens)) 
        vocab = {token:i for i, token in enumerate(tokens)} 
        return cls(vocab)            



if __name__ == "__main__":
    data_file = "cmn.txt"
    enc_data, dec_data = read_data(data_file)

    with open("encoder.json", "w", encoding="utf-8") as f:
        json.dump(enc_data, f)

    with open("decoder.json", "w", encoding="utf-8") as f:
        json.dump(dec_data, f)
   
    enc_vocab = Vocabulary.from_documents(enc_data)
    dec_vocab = Vocabulary.from_documents(dec_data)

    with open("cmn_vocab.bin", "bw") as f:
        pickle.dump((enc_vocab.vocab, dec_vocab.vo), f)

