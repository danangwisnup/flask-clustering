import os
import numpy as np
import flask
import pickle
from flask import Flask, redirect, url_for, request, render_template

# Membuat instance dari kelas
app = Flask(__name__, template_folder='templates')

# Membuat fungsi untuk memprediksi
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 4)
    loaded_model = pickle.load(open("./model/model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

# Route untuk halaman index
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')
    
# Route /result membawa data dari halaman index.html dengan metode POST
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        total_cases = request.form['total_cases']
        total_active_cases = request.form['total_active_cases']
        total_recovered = request.form['total_recovered']
        total_deaths = request.form['total_deaths']
        
        to_predict_list = list(map(float, [total_cases, total_active_cases, total_recovered, total_deaths]))
        result = ValuePredictor(to_predict_list)
        
        # Hasil clustering
        if float(result) == 0:
            clustering = 'Cluster 0: Jumlah kasus sedikit, jumlah kasus aktif sedikit, jumlah kasus sembuh sedikit, dan jumlah kasus meninggal sedikit.'
        elif float(result) == 1:
            clustering = 'Cluster 1: Jumlah kasus banyak, jumlah kasus aktif banyak, jumlah kasus sembuh banyak, dan jumlah kasus meninggal banyak.'
        else:
            clustering = 'Cluster 2: Jumlah kasus sedang, jumlah kasus aktif sedang, jumlah kasus sembuh sedang, dan jumlah kasus meninggal sedang.'

        return render_template("result.html", clustering=clustering)
