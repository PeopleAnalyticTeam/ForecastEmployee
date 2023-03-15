from inspect import isclass
import streamlit as st

st.set_page_config(page_title="", page_icon=":dart:") #, layout='wide')

"""## ":green[PREDIKSI DERET WAKTU]"\
"""
import pandas as pd
from darts import TimeSeries

from darts.models import ExponentialSmoothing

import matplotlib.pyplot as plt


#with st.echo('below'):
csv_data = st.file_uploader("Pilih file dengan format **:blue[.CSV]**")
delimiter = st.text_input("Tentukan sepataror file CSV", value=',', max_chars=1) #help='How your CSV values are separated')
if csv_data is None:
    st.warning("")
    st.stop()

custom_df = pd.read_csv(csv_data, sep=delimiter)
with st.expander("Tampilkan Nama Kolom dan Data"):
    st.dataframe(custom_df)

columns = list(custom_df.columns)
    
time_col = st.selectbox(":orange[Pilih Kolom Yang Berisi Keterangan Waktu]", columns) #help="Name of the column in your csv with time period data")
value_cols = st.selectbox(":orange[Pilih Kolom Yang Ingin Di Prediksi]", columns, 1) #, help="Name of column(s) with values to sample and forecast")
#options = {'Bulan': ('M', 12), 'Minggu': ('W', 52), 'Tahun': ('A', 1), 'Hari':  ('D', 365), 'Jam': ('H', 365 * 24), 'Kuarter': ('Q', 8)}
options = {'Monthly': ('M', 12), 'Weekly': ('W', 52), 'Yearly': ('A', 1), 'Daily':  ('D', 365), 'Hourly': ('H', 365 * 24), 'Quarterly': ('Q', 8)}
sampling_period = st.selectbox(":orange[Pilih Periode Deret Waktu Prediksi]", options) #, help='How to define samples. Pandas will sum entries between periods to create a well-formed Time Series')

custom_df[time_col] = pd.to_datetime(custom_df[time_col])
freq_string, periods_per_year = options[sampling_period]
custom_df = custom_df.set_index(time_col).resample(freq_string).sum()
with st.expander(":orange[Tampilkan Data Sesuai Periode Deret Waktu]"):
    st.write("Banyak Data Sesuai Periode Deret Waktu:", len(custom_df))
    st.dataframe(custom_df)

custom_series = TimeSeries.from_dataframe(custom_df, value_cols=value_cols)
st.subheader("Custom Training Controls")
max_periods = len(custom_series) - (2 * periods_per_year)
default_periods = min(10, max_periods)
num_periods = st.slider("Number of validation periods", key='cust_period', min_value=2, max_value=max_periods, value=default_periods, help='How many periods worth of datapoints to exclude from training')
num_samples = st.number_input("Number of prediction samples", key='cust_sample', min_value=1, max_value=10000, value=1000, help="Number of times a prediction is sampled for a probabilistic model")

st.subheader("Custom Plotting Controls")
low_quantile = st.slider('Lower Percentile', key='cust_low', min_value=0.01, max_value=0.99, value=0.05, help='The quantile to use for the lower bound of the plotted confidence interval.')
high_quantile = st.slider('High Percentile', key='cust_high', min_value=0.01, max_value=0.99, value=0.95, help='The quantile to use for the upper bound of the plotted confidence interval.')

train, val = custom_series[:-num_periods], custom_series[-num_periods:]
model = ExponentialSmoothing()
model.fit(train)
prediction = model.predict(len(val), num_samples=num_samples)

custom_fig = plt.figure()
custom_series.plot()

prediction.plot(label='forecast', low_quantile=low_quantile, high_quantile=high_quantile)

plt.legend()
st.pyplot(custom_fig)
