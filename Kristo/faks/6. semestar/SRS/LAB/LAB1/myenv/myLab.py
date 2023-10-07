from base64 import b64encode
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import scrypt
import sys
import json
import pickle

# Get the list of arguments passed to the script
args = sys.argv

if args[1] == 'init':
    # inicijalizacija baze, stvori novi .txt file
    # kreiraj salt, nonce i zapisi ih u prve dvije linije database.txt
    salt = get_random_bytes(16)
    nonce = get_random_bytes(16)

    with open('saltAndNonce.bin', 'wb') as f:
        f.write(salt)
        f.write(nonce)
    f.close()

    masterKey = args[2]
    key = scrypt(masterKey, salt, 32, N=2**14, r=8, p=1)
    data = b"ovdjeideurl:ovdjeidelozinka,hokus:pokus"

    cipher = AES.new(key, AES.MODE_GCM, nonce = nonce) 
    ciphertext, tag = cipher.encrypt_and_digest(data)

    with open('database.bin', 'wb') as f:
        f.write(ciphertext + tag)
        f.close()
    print("Password manager initialized.")

elif args[1] == 'put':
    # kodiranje url~password u bazu
    masterKey = args[2]
    url = args[3]
    password = args[4]
    oldSalt = ''
    oldNonce = ''

    with open('saltAndNonce.bin', 'rb') as f:
        linija = f.read()

        oldSalt = linija[:16]
        oldNonce = linija[16:]
        f.close()
    
    key = scrypt(masterKey, oldSalt, 32, N=2**14, r=8, p=1)
    suc = True
    exists = False
    first = True
    textToEncode = ''
    with open('database.bin', 'rb') as f:
        filedata = f.read().strip()
        
        cipher3 = AES.new(key, AES.MODE_GCM, nonce=oldNonce)
        ciphertext = filedata[:-16]
        tag = filedata[-16:]
        plaintext = b''
        try:
            plaintext = cipher3.decrypt_and_verify(ciphertext, tag)
            parovi = plaintext.decode('utf-8').strip().split(',')
            for line in parovi:
                plain = line.split(':')
                curl = plain[0]
                cpassword = plain[1]

            

                if curl == url:
                    exists = True
                    if first:
                        newpair = curl + ':' + password
                    else:
                        newpair = ',' + curl + ':' + password

                    textToEncode = textToEncode + newpair
                else:

                    if first:
                        textToEncode = textToEncode + curl + ':' + cpassword
                    else:
                        textToEncode = textToEncode + ',' + curl + ':' + cpassword
                
                first = False
            if(exists == False):
                textToEncode = textToEncode + ',' + url + ':' + password
        except ValueError as e:
            suc = False
            print("Master password incorrect or integrity check failed.")
    f.close()

    if suc == True:
        salt = get_random_bytes(16)
        nonce = get_random_bytes(16)
        key = scrypt(masterKey, salt, 32, N=2**14, r=8, p=1)
        with open('saltAndNonce.bin', 'wb') as f:
            f.write(salt)
            f.write(nonce)
        f.close()

        cipher = AES.new(key, AES.MODE_GCM, nonce = nonce) 
        ciphertext, tag = cipher.encrypt_and_digest(textToEncode.encode())

        with open('database.bin', 'wb') as f:
            f.write(ciphertext + tag)
            f.close()

        print("Stored password for " + url)

elif args[1] == 'get':
    getPassword = ''
    # dekodiranje url~password iz baze

    masterKey = args[2]
    url = args[3]
    oldSalt = ''
    oldNonce = ''
    suc = True

    with open('saltAndNonce.bin', 'rb') as f:
        linija = f.read()

        oldSalt = linija[:16]
        oldNonce = linija[16:]
        f.close()
    
    key = scrypt(masterKey, oldSalt, 32, N=2**14, r=8, p=1)
    
    exists = False
    first = True

    textToEncode = ''
    with open('database.bin', 'rb') as f:
        filedata = f.read().strip()
        
        cipher3 = AES.new(key, AES.MODE_GCM, nonce=oldNonce)
        ciphertext = filedata[:-16]
        tag = filedata[-16:]
        plaintext = b''
        try:
            plaintext = cipher3.decrypt_and_verify(ciphertext, tag)
            parovi = plaintext.decode('utf-8').strip().split(',')
            for line in parovi:
                plain = line.split(':')
                curl = plain[0]
                cpassword = plain[1]

            

                if curl == url:
                    exists = True
                    getPassword = cpassword
                    if first:
                        newpair = curl + ':' + cpassword
                    else:
                        newpair = ',' + curl + ':' + cpassword

                    textToEncode = textToEncode + newpair
                else:

                    if first:
                        textToEncode = textToEncode + curl + ':' + cpassword
                    else:
                        textToEncode = textToEncode + ',' + curl + ':' + cpassword
                
                first = False
            if(exists == False):
                print("URL not found")
        except ValueError as e:
            suc = False
            print("Master password incorrect or integrity check failed.")
    f.close()

    if suc == True:
        salt = get_random_bytes(16)
        nonce = get_random_bytes(16)
        key = scrypt(masterKey, salt, 32, N=2**14, r=8, p=1)
        with open('saltAndNonce.bin', 'wb') as f:
            f.write(salt)
            f.write(nonce)
        f.close()

        cipher = AES.new(key, AES.MODE_GCM, nonce = nonce) 
        ciphertext, tag = cipher.encrypt_and_digest(textToEncode.encode())

        with open('database.bin', 'wb') as f:
            f.write(ciphertext + tag)
            f.close()

        print("Password for " + url + " is : " + getPassword)


else:
    print("Wrong use of program: init|get|put url masterKey")
