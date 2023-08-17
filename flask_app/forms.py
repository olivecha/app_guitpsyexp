from flask_wtf import FlaskForm
from wtforms import (StringField, TextAreaField, IntegerField, BooleanField,
                     RadioField)
from wtforms.validators import InputRequired, Length

class SoundChoice(FlaskForm):
    """
    FlaskForm inherited class to manage the favorite sound choice and the user ability
    """
    sounds = RadioField('Son préféré :',
                        choices = [('A', 'Son A'), ('B', 'Son B')],
                        validators = [InputRequired()])

    ability = RadioField('Auto-évaluation de l\'oreille musicale : ',
                        choices = [('1', 'Débutant'), ('2', 'Intermédiaire'), ('3', 'Professionel')],
                        validators = [InputRequired()])

