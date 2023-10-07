from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from weatherApi import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(64), nullable=False)
    password = db.Column(db.String(64), nullable=False)
    def_location = db.Column(db.String(64))


class Location(db.Model):
    email = db.Column(db.String(120), db.ForeignKey('user.email'), primary_key=True)
    location = db.Column(db.String(64))


# @login.user_loader
# def load_user(id):
#     return User.query.get(int(id))
