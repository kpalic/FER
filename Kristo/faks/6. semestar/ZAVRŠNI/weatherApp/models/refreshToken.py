from database import db

class RefreshToken(db.Model):
    __tablename__ = 'refresh_tokens'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    token = db.Column(db.String, nullable=False)
    user_agent = db.Column(db.String)
    ip = db.Column(db.String)
    expires = db.Column(db.DateTime, nullable=False)
