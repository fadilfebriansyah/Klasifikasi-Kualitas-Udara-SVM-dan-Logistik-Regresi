import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import mysql.connector
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
import base64


app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['ENV'] = 'development'
# Meload model dari file pickle
with open('udara_svm.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


# Konfigurasi koneksi ke MySQL
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'quality_udara',
}

# Membuat koneksi ke MySQL
conn = mysql.connector.connect(**db_config)

# Mapping of prediction classes to labels
class_labels = {
    0: 'SANGAT TIDAK SEHAT',
    1: 'TIDAK SEHAT',
    2: 'SEDANG',
    3: 'BAIK',
    # Add more labels as needed
}

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Extracting the date from the form
    tanggal = request.form.get('tanggal')
    # Convert the date string to a datetime object (if needed)
    # tanggal = datetime.strptime(tanggal, '%Y-%m-%d')

    # Extracting other features from the form
    float_features = [float(request.form[x]) for x in ['pm10', 'so2', 'co', 'o3', 'no2', 'max']]
    features = [np.array(float_features)]
    print("Input Features:", features)
    prediction = model.predict(features)[0]
    print("Prediction:", prediction)
    prediction_label = class_labels.get(prediction, 'Unknown')
    print("Prediction Label:", prediction_label) 
    print("Insert Data:", float_features + [prediction_label])

    try:
        cursor = conn.cursor()
        insert_query = "INSERT INTO datanew (tanggal, pm10, so2, co, o3, no2, max, categori) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        insert_data = (tanggal,) + tuple(float_features + [prediction_label])
        cursor.execute(insert_query, insert_data)
        conn.commit()
        cursor.close()
    except Exception as e:
        print(f"Error during data insertion: {str(e)}")

    return render_template("index.html", prediction_text=f"{prediction_label}")

def generate_pie_chart():
    cursor = conn.cursor()
    query = "SELECT categori, COUNT(*) FROM datanew GROUP BY categori"
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()

    labels, sizes = zip(*data)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Categories')

    # Save the pie chart to a BytesIO object
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    plt.close()

    # Convert the image to base64 for embedding in HTML
    image_data = base64.b64encode(image_stream.getvalue()).decode('utf-8')
    return image_data

@app.route('/piechart')
def show_chart():
    image_data = generate_pie_chart()
    return render_template('piechart.html', image_data=image_data)

def generate_bar_chart():
    # Connect to the database
    cursor = conn.cursor()

    # SQL query to get the data
    query = """
        SELECT YEAR(tanggal) AS tahun, categori, COUNT(*) AS jumlah
        FROM datanew
        GROUP BY tahun, categori
    """

    # Execute the query and fetch the data
    cursor.execute(query)
    data = cursor.fetchall()

    # Close the database connection
    cursor.close()

    # Create a DataFrame from the fetched data
    df = pd.DataFrame(data, columns=['tahun', 'categori', 'jumlah'])

    # Pivot the DataFrame for plotting
    pivot_df = df.pivot(index='tahun', columns='categori', values='jumlah').fillna(0)

    # Create a bar chart with a larger figure size
    plt.figure(figsize=(10, 6))
    ax = pivot_df.plot(kind='bar')

    # Set chart title and axis labels
    plt.title('Jumlah Kategori Kualitas Udara per Tahun')
    plt.xlabel('Tahun')
    plt.ylabel('Jumlah')

    # Move the legend to the upper right corner and adjust the bbox_to_anchor to avoid cutting off
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), title='Kategori', fontsize='small', ncol=4)

    # Add space between the legend and xlabel
    plt.subplots_adjust(bottom=0.2)

    # Save the bar chart to a BytesIO object
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png', bbox_inches='tight')

    # Close the plot to free up resources
    plt.close()

    # Convert the image to base64 for embedding in HTML
    image_data = base64.b64encode(image_stream.getvalue()).decode('utf-8')

    return image_data

@app.route('/barchart')
def show_barchart():
    image_data = generate_bar_chart()
    return render_template('barchart.html', image_data=image_data)

if __name__ == '__main__':
    app.run(debug=True)
    app.run(port=5001)
    conn.close()  # Menutup koneksi ke MySQL setelah aplikasi berakhir
