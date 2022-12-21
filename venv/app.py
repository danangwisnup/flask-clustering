import os
import numpy as np
import pandas as pd
import flask
import pickle
from flask import Flask, redirect, url_for, request, render_template

# Membuat instance dari kelas
app = Flask(__name__, template_folder='templates')

# Membuat fungsi untuk memprediksi
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 3)
    loaded_model = pickle.load(open("./model/kmeans.pkl", "rb"))
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
        population_density = request.form['population_density']
        mortality = request.form['mortality']
        
        to_predict_list = list(map(float, [total_cases, population_density, mortality]))
        result = ValuePredictor(to_predict_list)

        # read data.csv
        data = pd.read_csv('./model/data.csv')

        # Menampilkan hasil clustering
        if float(result) == 0:
            clustering = 'Cluster 1'
            Ctotal_cases = data[data['K-means'] == 0]['Total Cases'].mean()
            Cpopulation_density = data[data['K-means'] == 0]['Population Density'].mean()
            Cmortality = data[data['K-means'] == 0]['Mortality'].mean()
            Ctotal_deaths = data[data['K-means'] == 0]['Total Deaths'].mean()
            Ctotal_recovered = data[data['K-means'] == 0]['Total Recovered'].mean()
            Ctotal_active_cases = data[data['K-means'] == 0]['Total Active Cases'].mean()
            Cprovince = data[data['K-means'] == 0]['Province'].values
            Koment = "Klaster ini memiliki total kasus yang tergolong sedikit namun persentase tingkat kematiannya cukup banyak, dalam cluster ini perlu ditingkatkan terutama kinerja tenaga medis dan semangat pasien dalam menangani kasus COVID-19 serta tindakan-tindakan preventif yang dapat dilakukan dari masing-masing individu sesuai dengan protokol kesehatan yang ada."
        elif float(result) == 1:
            clustering = 'Cluster 2'
            Ctotal_cases = data[data['K-means'] == 1]['Total Cases'].mean()
            Cpopulation_density = data[data['K-means'] == 1]['Population Density'].mean()
            Cmortality = data[data['K-means'] == 1]['Mortality'].mean()
            Ctotal_deaths = data[data['K-means'] == 1]['Total Deaths'].mean()
            Ctotal_recovered = data[data['K-means'] == 1]['Total Recovered'].mean()
            Ctotal_active_cases = data[data['K-means'] == 1]['Total Active Cases'].mean()
            Cprovince = data[data['K-means'] == 1]['Province'].values
            Koment = "Klaster ini memiliki persentase tingkat kematian yang cukup tinggi jika dilihat dari total kasus tergolong rendah maka perlu ditingkatkan kinerja dalam penanganan COVID-19 baik dari tim tenaga medis maupun pasien COVID-19 nya."
        elif float(result) == 2:
            clustering = 'Cluster 3'
            Ctotal_cases = data[data['K-means'] == 2]['Total Cases'].mean()
            Cpopulation_density = data[data['K-means'] == 2]['Population Density'].mean()
            Cmortality = data[data['K-means'] == 2]['Mortality'].mean()
            Ctotal_deaths = data[data['K-means'] == 2]['Total Deaths'].mean()
            Ctotal_recovered = data[data['K-means'] == 2]['Total Recovered'].mean()
            Ctotal_active_cases = data[data['K-means'] == 2]['Total Active Cases'].mean()
            Cprovince = data[data['K-means'] == 2]['Province'].values
            Koment = "Klaster ini memiliki total kasus paling dan tingkat kematian paling tinggi maka perlu lebih ditingkatkan kembali kesadaran dari tiap individu-nya dalam menerapkan protokol Kesehatan dan melakukan berbagai tindakan preventif pencegahan COVID-19 lainnya, namun jika dilihat dari persentase kemungkinan kematian tidak berbanding jauh dengan Klaster 6 yang memiliki total kasus yang lebih sedikit, maka bisa disimpulkan bahwa tenaga medis dan pasien di kluster 2 sudah cukup baik dalam menangani kasus COVID-19."
        elif float(result) == 3:
            clustering = 'Cluster 4'
            Ctotal_cases = data[data['K-means'] == 3]['Total Cases'].mean()
            Cpopulation_density = data[data['K-means'] == 3]['Population Density'].mean()
            Cmortality = data[data['K-means'] == 3]['Mortality'].mean()
            Ctotal_deaths = data[data['K-means'] == 3]['Total Deaths'].mean()
            Ctotal_recovered = data[data['K-means'] == 3]['Total Recovered'].mean()
            Ctotal_active_cases = data[data['K-means'] == 3]['Total Active Cases'].mean()
            Cprovince = data[data['K-means'] == 3]['Province'].values
            Koment = "Klaster ini memiliki persentase tingkat kematian yang cukup rendah dan total kasus juga rendah maka Tindakan yang perlu dilakukan dalam cluster ini yaitu tindakan preventif dalam pencegahan COVID-19."
        elif float(result) == 4:
            clustering = 'Cluster 5'
            Ctotal_cases = data[data['K-means'] == 4]['Total Cases'].mean()
            Cpopulation_density = data[data['K-means'] == 4]['Population Density'].mean()
            Cmortality = data[data['K-means'] == 4]['Mortality'].mean()
            Ctotal_deaths = data[data['K-means'] == 4]['Total Deaths'].mean()
            Ctotal_recovered = data[data['K-means'] == 4]['Total Recovered'].mean()
            Ctotal_active_cases = data[data['K-means'] == 4]['Total Active Cases'].mean()
            Cprovince = data[data['K-means'] == 4]['Province'].values
            Koment = "Klaster ini memiliki persentase tingkat kematian yang sedang jika dilihat dari total kasus tergolong tinggi maka perlu ditingkatkan kinerja dalam penanganan COVID-19 baik dari tim tenaga medis maupun pasien COVID-19 nya."
        elif float(result) == 5:
            clustering = 'Cluster 6'
            Ctotal_cases = data[data['K-means'] == 5]['Total Cases'].mean()
            Cpopulation_density = data[data['K-means'] == 5]['Population Density'].mean()
            Cmortality = data[data['K-means'] == 5]['Mortality'].mean()
            Ctotal_deaths = data[data['K-means'] == 5]['Total Deaths'].mean()
            Ctotal_recovered = data[data['K-means'] == 5]['Total Recovered'].mean()
            Cprovince = data[data['K-means'] == 5]['Province'].values
            Koment = "Klaster ini memiliki total kasus paling dan tingkat kematian ke-2 tertinggi maka perlu lebih ditingkatkan kembali kesadaran dari tiap individu-nya dalam menerapkan protokol Kesehatan dan melakukan berbagai tindakan preventif pencegahan COVID-19 lainnya, namun jika dilihat dari persentase kemungkinan kematian tidak berbanding jauh dengan Klaster 2 yang memiliki total kasus jauh lebih banyak bahkan 4x nya, namun persentase kemungkinan kematian nya tidak berbanding jauh maka bisa disimpulkan bahwa penanganan tenaga medis dan pasien COVID-19 perlu ditingkatkan kembali."
        
        # Mengubah tipe data array menjadi string
        Cprovince = ', '.join(Cprovince)

        return render_template(
            "result.html", 
            clustering=clustering, 
            total_cases=total_cases, 
            population_density=population_density, 
            mortality=mortality,
            Ctotal_cases=Ctotal_cases,
            Cpopulation_density=Cpopulation_density,
            Cmortality=Cmortality,
            Ctotal_deaths=Ctotal_deaths,
            Ctotal_recovered=Ctotal_recovered,
            Ctotal_active_cases=Ctotal_active_cases,
            Cprovince=Cprovince,
            Koment=Koment
            )
