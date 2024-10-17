from vocab import Vocab

# Primjer frekvencija za testiranje
frequencies = {'the': 10, 'a': 8, 'an': 5, 'dog': 4, 'cat': 4, 'elephant': 2}

# Kreiranje vokabulara s minimalnom frekvencijom 1 i neograničenom veličinom
vocab = Vocab(frequencies, max_size=-1, min_freq=1)

# Prikaz itos i stoi
print(f"itos: {vocab.itos}")
print(f"stoi: {vocab.stoi}")

# Kodiranje i dekodiranje riječi
tokens = ['the', 'cat', 'in', 'the', 'hat']
encoded_tokens = vocab.encode(tokens)
decoded_tokens = vocab.decode(encoded_tokens.tolist())
print(f"Tokens: {tokens}")
print(f"Encoded: {encoded_tokens}")
print(f"Decoded: {decoded_tokens}")

# Kodiranje riječi koje nije u vokabularu
unknown_token = 'unknown'
encoded_unknown = vocab.encode(unknown_token)
print(f"Encoded unknown token: {encoded_unknown}")
print(f"Decoded unknown token: {vocab.decode(encoded_unknown.item())}")
