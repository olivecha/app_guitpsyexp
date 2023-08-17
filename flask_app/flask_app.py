from flask import Flask, render_template, send_from_directory, request, redirect, url_for, flash
from forms import SoundChoice
import sqlite3
import numpy as np
import sys
sys.path.append('..')
from soundgen.generate import random_envelop_parameter, generate_chord, save_wav
app = Flask(__name__)
app.config['SECRET_KEY'] = 'development'
PASSPHRASE = 'bruand123'


# Main web application
@app.route('/', methods=["GET", "POST"])
def index():   
    pA, pB = make_new_random_sounds()
    
    form = SoundChoice()
    message = ''

    if request.method == "POST":        
        if form.sounds.data is not None and form.ability.data is not None:
            conn = get_db_connection()
            conn.execute('INSERT INTO soundstats (env_param_A, env_param_B, fav_sound, ability) VALUES (?, ?, ?, ?)',
                         (pA, pB, form.sounds.data, form.ability.data))

            conn.commit()
            conn.close()

            return redirect(url_for('entry_completed'))

        elif form.sounds.data is  not None:
            message = 'Veuillez sélectionner un niveau de compétence musicale, attention les sons changent à chaque chargement de page'
        else :
            message = 'Veuillez sélectionner un son, attention les sons changent à chaque chargement de page'

    return render_template('index.html', 
                           message=message, 
                           param1=np.around(pA, 3), 
                           param2=np.around(pB, 3), 
                           form=form)


@app.route('/success')
def entry_completed():
    return redirect(url_for('index'))

@app.route('/about')
def about():
    return render_template('about.html')

def password_prompt():
    return render_template('password.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    """
    Result page for the psychoacoustic experience
    """
    if request.method == 'GET':
        return password_prompt()
    elif request.method == 'POST':
        if request.form['password'] != PASSPHRASE:
            return password_prompt("Invalid password, try again. Admin password:")
        else:
            mean_favorite, sample_size, users = compute_results()
            return render_template('results.html', 
                           mean_favorite=mean_favorite,
                           sample_size=sample_size,
                           users=users)

def compute_results():
    """ Compute results from the database """
    # Fetch all the database data
    conn = get_db_connection()
    data = conn.execute('SELECT * FROM soundstats').fetchall()
    conn.close()
    # Get the envelop values for all the chosen sounds
    idx_key = {'A':1, 'B':2}
    all_favorites = []
    for dpoint in data:
        all_favorites.append(dpoint[idx_key[dpoint[3]]])
    # Favorite is the mean
    mean_favorite = np.mean(all_favorites)
    # Get how many people answered for fun
    sample_size = len(all_favorites) 
    # Get the user data
    user_keys = {1:'deb',
                 2:'int',
                 3:'pro' }
    users = {'deb':0,
             'int':0,
             'pro':0 }
    # Compile all the users
    for dpoint in data:
        users[user_keys[dpoint[4]]] += 1 
    return mean_favorite, sample_size, users


def make_new_random_sounds():
    """
    Generate two new random chord sounds and save them in assets/temp
    """
    # Generate two random envelop parameters
    p1 = random_envelop_parameter(bounds=(-5, 5))
    p2 = random_envelop_parameter(bounds=(-5, 5))
    # Generate two sounds from the envelop parameters
    chord1 = generate_chord(p1) 
    chord2 = generate_chord(p2) 
    # Save two sounds to temp files
    save_wav('static/temp_sound1', chord1)
    save_wav('static/temp_sound2', chord2)
    # Return the radom envelop parameters to save to a dataframe
    return p1, p2


def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn
