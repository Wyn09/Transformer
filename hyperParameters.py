import torch
def getHyperParameters():
    epoch = 1000
    learning_rate = 1.5e-4
    batch_size = 8
    emb_size = 128
    num_layers = 4
    num_heads = 8
    dropout = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return epoch, learning_rate, batch_size, emb_size, num_layers, num_heads, dropout, device