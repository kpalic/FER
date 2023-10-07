from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired

from flask import Blueprint, redirect, session
from flask_jwt_extended import jwt_required, get_jwt_identity

from models.location import Location
from database import db

location_bp = Blueprint('location', __name__)

class LocationForm(FlaskForm):
    email = StringField('E-mail', validators=[DataRequired()])
    location = StringField('Location', validators=[DataRequired()])

@location_bp.route('/location', methods=['POST'])
@jwt_required() 
def location():
    # data = request.get_json()
    # print(data)
    current_user = get_jwt_identity()
    form = LocationForm()
    location = Location(email=form.email.data,
                        location=form.location.data)
    db.session.add(location)
    db.session.commit()
    print('Added new location ', location.location, ' for user ', location.email)
    # else:
    #     error = form.errors
    #     print(error)

    return redirect('/')
