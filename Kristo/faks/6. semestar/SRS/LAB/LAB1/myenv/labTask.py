import json
from base64 import b64encode
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import sys

# Get the list of arguments passed to the script
args = sys.argv

# TODO: Implement database initialization logic
if args[1] == 'init':
    with open('database.txt', 'w') as f:
        f.write('')

# TODO: Implement password storage logic
elif args[1] == 'put':
    # Get the key, url, and password from the command line arguments
    key = args[2]
    url = args[3]
    password = args[4]

    # Define the header and data
    header = url
    data = password

    # Generate a random 16-byte key
    nonce = get_random_bytes(16)

    # Create an AES-GCM cipher object using the key
    cipher = AES.new(key, AES.MODE_GCM, nonce)

    # Update the cipher with the header
    cipher.update(header)

    # Encrypt the data and get the ciphertext and tag
    ciphertext, tag = cipher.encrypt_and_digest(data)

    # Create a JSON object containing the encrypted data
    json_k = ['nonce', 'header', 'ciphertext', 'tag']
    json_v = [b64encode(x).decode('utf-8') for x in (cipher.nonce, cipher.header, ciphertext, tag)]
    result = json.dumps(dict(zip(json_k, json_v)))

    # Write the encrypted data to the database file
    with open('database.txt', 'a') as f:
        f.write(result + '\n')

# TODO: Implement password retrieval logic
elif args[1] == 'get':
    # Get the key and url from the command line arguments
    key = args[2]
    url= args[3]

    # Read the encrypted data from the database file
    with open('database.txt', 'r') as f:
        for line in f:
            # Parse the JSON object
            data = json.loads(line.strip())

            # Get the header and ciphertext from the JSON object
            header = b64decode(data['header'])
            ciphertext = b64decode(data['ciphertext'])

            jv = {k:b64encode(b64[k]) for k in json_k}

            cipher = AES.new(key, AES.MODE_GCM, nonce = jv['nonce'])
            cipher.update(jv['header'])
            plaintext = cipher.decrypt_and_verify(jv['ciphertext'], jv['tag'])

            if(url == )
            print("The message was: " + plaintext.decode('utf-8'))

            # Create an AES-GCM cipher object using the key and nonce
            cipher = AES.new(key, AES.MODE_GCM, nonce=b64decode(data['nonce']))

            # Update the cipher with the header
            cipher.update(header)

            # Decrypt the ciphertext to get the plaintext
            plaintext = cipher.decrypt(ciphertext)

            # Check the tag to verify the authenticity of the plaintext
            try:
                cipher.verify(b64decode(data['tag']))
                print(plaintext.decode('utf-8'))
                break
            except ValueError:
                continue

