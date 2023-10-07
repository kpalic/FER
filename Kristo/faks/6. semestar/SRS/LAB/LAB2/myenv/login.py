import sys
import getpass
import json
import hashlib
import time
import os

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f)

def hash_password(password, salt):
    sha256 = hashlib.sha256()
    sha256.update((password + salt).encode("utf-8"))
    return sha256.hexdigest()

def load_users():
    try:
        with open("users.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def login(username):
    users = load_users()
    hashed_username = hashlib.sha256(username.encode("utf-8")).hexdigest()

    if hashed_username not in users:
        return False

    user = users[hashed_username]

    password = getpass.getpass("Password: ")
    hashed_password = hash_password(password, user["salt"])
    if hashed_password != user["password"]:
        for i in range(10):
            time.sleep(1)
            print(i)
        return False

    if user["forcepass"]:
        new_password = getpass.getpass("New password: ")
        confirm_password = getpass.getpass("Repeat new password: ")

        if new_password == password:
            print("Password change failed. New password must be different from the old password.")
            return False

        if new_password != confirm_password:
            print("Password change failed. Password mismatch.")
            return False

        salt = os.urandom(16).hex()
        hashed_new_password = hash_password(new_password, salt)
        user["password"] = hashed_new_password
        user["salt"] = salt
        user["forcepass"] = False
        save_users(users)
        print("Password changed successfully.")

    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: login.py <username>")
        sys.exit(1)

    username = sys.argv[1]

    if login(username):
        print("Login successful.")
    else:
        print("Username or password incorrect.")
