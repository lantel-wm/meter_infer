import mysql.connector
import plotly.graph_objects as go
import pandas as pd
import time

db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '1AlgorithM',
    'database': 'meter_readings'
}

conn = mysql.connector.connect(**db_config)

fig = go.Figure()
fig.add_trace(go.Scatter(x=[], y=[], name='Instrument 1'))
fig.show()

while True:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Readings WHERE instrument_id = 1")
    data = cursor.fetchall()
    cursor.close()

    df = pd.DataFrame(data, columns=['reading_id', 'camera_id', 'instrument_id', 'value', 'datetime', 'rect_x', 'rect_y', 'rect_h', 'rect_w', 'debug_image_path'])

    fig.update_traces(x=df['datetime'], y=df['value'])

    fig.update_layout(title='Meter Readings', xaxis_title='Time', yaxis_title='Value')
    

    time.sleep(1)