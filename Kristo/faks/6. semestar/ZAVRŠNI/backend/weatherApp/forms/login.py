from flask import Blueprint, redirect, jsonify, request
from flask_wtf import FlaskForm
from flask_cors import cross_origin
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_required, get_jwt_identity
from wtforms import StringField, PasswordField
from wtforms.validators import DataRequired, ValidationError

from config import Config
from models.user import User
from models.refreshToken import RefreshToken
from database import db

import datetime
import jwt

login_bp = Blueprint('login', __name__)

class LoginForm(FlaskForm):
    email = StringField('E-mail', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])

@login_bp.route('/login', methods=['POST'])
@cross_origin(origin='localhost:5000/')
def login():
    form = LoginForm()
    user = User.query.filter_by(email=form.email.data).first()

    if user is None or user.password != form.password.data:
        print("Error")
        return jsonify({'status': 'error', 'message': 'Incorrect email or password'})
    else:
        # Kreiranje access i refresh tokena
        access_token = create_access_token(identity=user.email)
        
        # Generiranje payloada za refresh token
        expires = datetime.datetime.utcnow() + datetime.timedelta(days=30)
        payload = {
            'user_id': user.id,
            'exp': expires
        }
        refresh_token = jwt.encode(payload, Config.JWT_SECRET_KEY, algorithm='HS256')

        # Pohranjivanje refresh tokena u bazu podataka
        ip = request.remote_addr
        user_agent = request.headers.get('User-Agent')
        token_entry = RefreshToken(user_id=user.id, token=refresh_token, ip=ip, user_agent=user_agent, expires=expires)
        db.session.add(token_entry)
        db.session.commit()

        print("Successful login for ", user.username)
        return jsonify({'status': 'success', 'message': 'Successful login', 'access_token': access_token, 'refresh_token': refresh_token})

# def login():
#     form = LoginForm()
#     user = User.query.filter_by(email=form.email.data).first()
#     # create token
#     if user is None or user.password != form.password.data:
#         print("Error")
#         return jsonify({'status': 'error', 'message': 'Incorrect email or password'})
#     else:
#         access_token = create_access_token(identity=user.email)
#         refresh_token = create_refresh_token(identity=user.email)
#         print("Successfull login for ", user.username)
#         return jsonify({'status': 'success', 'message': 'Successful login'})


@login_bp.route('/logout', methods=['POST'])
def logout():
    print("eto nas ovdje")
    data = request.get_json()
    token = data.get('refreshToken')
    print(data)
    refresh_token = RefreshToken.query.filter_by(token=token).first()
    if refresh_token is None:
        return jsonify({'error': 'Invalid token'}), 401
    db.session.delete(refresh_token)
    db.session.commit()
    return jsonify({'message': 'Logout successful'})

from flask import current_app

@login_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    ip = request.remote_addr
    user_agent = request.headers.get('User-Agent')
    current_user = get_jwt_identity()

    # Dobivanje korisnika iz baze podataka koristeći email
    user = User.query.filter_by(email=current_user['email']).first()

    # Ako korisnik ne postoji, vratite odgovarajuću poruku o pogrešci
    if not user:
        return jsonify({'message': 'User not found'}), 404

    # Dobivanje user id-a iz korisnika
    user_id = user.id

    # Kreiranje novog pristupnog tokena
    new_token = create_access_token(identity=current_user)

    # Postavljanje datuma isteka za refresh token
    expires = datetime.datetime.utcnow() + datetime.timedelta(days=30)

    # Generiranje novog refresh tokena
    payload = {
        'user_id': user_id,
        'exp': expires
    }
    token = jwt.encode(payload, current_app.config['SECRET_KEY'], algorithm='HS256').decode('utf-8')
    
    # Dodavanje novog refresh tokena u bazu podataka
    refresh_token = RefreshToken(user_id=user_id, token=token, ip=ip, user_agent=user_agent, expires=expires)
    db.session.add(refresh_token)
    db.session.commit()

    # Povratak pristupnog tokena
    return jsonify({'access_token': new_token})


