import sys
import json
import getpass
import hashlib
import os
import time
import re

def hash_password(password, salt):
    sha256 = hashlib.sha256()
    sha256.update((password + salt).encode("utf-8"))
    return sha256.hexdigest()

def generate_salt():
    return os.urandom(16).hex()

def load_users():
    try:
        with open("users.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f)

def is_complex(password):
    if len(password) < 8 or len(password) > 20:
        return False
    if not re.search(r"[a-zA-Z]{1,19}", password):
        return False
    if not re.search(r"\d", password):
        return False
    return True


def add_user(username):
    users = load_users()
    hashed_username = hashlib.sha256(username.encode("utf-8")).hexdigest()
    if hashed_username in users:
        print("User already exists.")
        return
    password = getpass.getpass("Password: ")
    confirm_password = getpass.getpass("Repeat Password: ")

    if password != confirm_password:
        print("User add failed. Password mismatch.")
        return
    if not is_complex(password):
        print("User add failed. Password must be at least 8 characters long and contain at least one number.")
        return

    salt = generate_salt()
    hashed_password = hash_password(password, salt)
    users[hashed_username] = {
        "password": hashed_password,
        "salt": salt,
        "forcepass": False
    }
    save_users(users)
    print(f"User {username} successfully added.")

def change_password(username):
    users = load_users()
    hashed_username = hashlib.sha256(username.encode("utf-8")).hexdigest()
    if hashed_username not in users:
        print("User not found.")
        return

    password = getpass.getpass("Password: ")
    confirm_password = getpass.getpass("Repeat Password: ")

    if password != confirm_password:
        print("Password change failed. Password mismatch.")
        return
    if not is_complex(password):
        print("Password change failed. Password must be at least 8 characters long and contain at least one number.")
        return

    salt = generate_salt()
    hashed_password = hash_password(password, salt)
    users[hashed_username]["password"] = hashed_password
    users[hashed_username]["salt"] = salt
    save_users(users)
    print("Password change successful.")

def force_password_change(username):
    users = load_users()
    hashed_username = hashlib.sha256(username.encode("utf-8")).hexdigest()
    if hashed_username not in users:
        print("User not found.")
        return

    users[hashed_username]["forcepass"] = True
    save_users(users)
    print(f"User {username} will be requested to change password on next login.")

def delete_user(username):
    users = load_users()
    hashed_username = hashlib.sha256(username.encode("utf-8")).hexdigest()
    if hashed_username not in users:
        print("User not found.")
        return

    del users[hashed_username]
    save_users(users)
    print(f"User {username} successfully removed.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: usermgmt.py <operation> <username>")
        sys.exit(1)

    operation = sys.argv[1]
    username = sys.argv[2]

    if operation == "add":
        add_user(username)
    elif operation == "passwd":
        change_password(username)
    elif operation == "forcepass":
        force_password_change(username)
    elif operation == "del":
        delete_user(username)
    else:
        print("Invalid operation.")
        sys.exit(1)

