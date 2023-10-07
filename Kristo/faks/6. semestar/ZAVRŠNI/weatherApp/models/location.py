from database import db

class Location(db.Model):
    __tablename__ = 'locations'
    email = db.Column(db.String(120), db.ForeignKey('users.email'), primary_key=True)
    location = db.Column(db.String(64))
