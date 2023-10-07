import json
from base64 import b64encode
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import sys

# Get the list of arguments passed to the script
args = sys.argv

header = args[1]

data = args[2]

print(header, data)

# Print the arguments
#print(args[1])

# definiranje varijabli
header = b"header"
data = b"secret"

# generiranje 16-bitnog kljuca
key = get_random_bytes(16)

# instanca aes-gcm ciphera se kreira koristeci generirani kljuc
cipher = AES.new(key, AES.MODE_GCM)

# azuriranje headera
cipher.update(header)

# generirani su ciphertext i tag
# methoda encrypt_and_digest nad cipher objektom sifrira data varijablu
# i vraca ciphertext i tag
ciphertext, tag = cipher.encrypt_and_digest(data)

# kreiranje JSON objekta koji sadrzi enkriptirane podatke
# json_k - keys
# json_v - values - Base64 enkriptirane vrijednosti od cipher.nonce, header, ciphertext i tag
json_k = [ 'nonce', 'header', 'ciphertext', 'tag' ]
json_v = [ b64encode(x).decode('utf-8') for x in (cipher.nonce, header, ciphertext, tag) ]

# zip() kombinira json_k i json_v liste i kreira dictionary
# json.dumps() serijalizira dictionary u string koji se printa
# output je JSON string Base64 enkodirana vrijednost za :
# nonce header ciphertext i tag.
result = json.dumps(dict(zip(json_k, json_v)))
print(result)
{"nonce": "DpOK8NIOuSOQlTq+BphKWw==", "header": "aGVhZGVy", "ciphertext": "CZVqyacc", "tag": "B2tBgICbyw+Wji9KpLVa8w=="}