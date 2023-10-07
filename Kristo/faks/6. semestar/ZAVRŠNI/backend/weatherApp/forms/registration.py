from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, ValidationError, EqualTo

from flask import Blueprint, request, jsonify, redirect, session
from flask_login import current_user, login_user, logout_user
import requests

from config import Config
from models.user import User
from database import db

registration_bp = Blueprint('register', __name__)

class RegistrationForm(FlaskForm):
    email = StringField('E-mail', validators=[DataRequired()])
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    def_location = StringField('location')

    # def validate_username(self, username):
    #     user = User.query.filter_by(username=username.data).first()
    #     if user is not None:
    #         raise ValidationError('Please use a different username.')

    def validate_email(self, email):
        user = User.query.filter_by(email=self.email.data).first()
        if user is not None:
            return False

@registration_bp.route('/register', methods=['POST'])
def register():
    form = RegistrationForm()
    if form.validate():
        user = User(email=form.email.data,
                    username=form.username.data,
                    password=form.password.data,
                    def_location=form.def_location.data)
        db.session.add(user)
        db.session.commit()
        print('Congratulations, you are now a registered user!')
        return jsonify({'status': 'Okay', 'message': 'Successful registration.'})
    else:
        if user is not None:
            print("Error")
            return jsonify({'status': 'error', 'message': 'E-mail already in use.'})
