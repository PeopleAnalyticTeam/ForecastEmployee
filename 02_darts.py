from inspect import isclass
import streamlit as st
st.set_page_config(page_title='PREDIKSI JUMLAH KARYAWAN', page_icon=":dart:", layout='wide')

"""## UPLOAD DATA YANG AKAN DI PREDIKSI!\
"""

import darts.datasets as ds
all_datasets = {x: ds.__getattribute__(x) for x in dir(ds) if isclass(ds.__getattribute__(x)) and x not in ("DatasetLoaderMetadata", "DatasetLoaderCSV")}
with st.expander("More info on Darts Datasets"):
    for name, dataset in all_datasets.items():
        st.write(f"#### {name}\n\n{dataset.__doc__}")

with st.echo('below'):
    use_example = st.checkbox("Use example dataset")
    if use_example:
        dataset_choice = st.selectbox("Example Dart Dataset", all_datasets, index=5)
        with st.spinner("Fetching Dataset"):
            dataset = all_datasets[dataset_choice]()
            timeseries = dataset.load()
            custom_df = timeseries.pd_dataframe()
            custom_df['Period'] = custom_df.index.to_series()
            custom_df = custom_df[['Period', *custom_df.columns[:-1]]]
    else:
        csv_data = st.file_uploader("New Timeseries csv")
        delimiter = st.text_input("CSV Delimiter", value=',', max_chars=1, help='How your CSV values are separated')

        if csv_data is None:
            st.warning("Upload a CSV to analyze")
            st.stop()

        custom_df = pd.read_csv(csv_data, sep=delimiter)
    with st.expander("Show Raw Data"):
        st.dataframe(custom_df)

    columns = list(custom_df.columns)
    with st.expander("Show all columns"):
        st.write(' | '.join(columns))

    time_col = st.selectbox("Time Column", columns, help="Name of the column in your csv with time period data")
    value_cols = st.selectbox("Values Column(s)", columns, 1, help="Name of column(s) with values to sample and forecast")
    options = {'Monthly': ('M', 12), 'Weekly': ('W', 52), 'Yearly': ('A', 1), 'Daily':  ('D', 365), 'Hourly': ('H', 365 * 24), 'Quarterly': ('Q', 8)}
    sampling_period = st.selectbox("Time Series Period", options, help='How to define samples. Pandas will sum entries between periods to create a well-formed Time Series')

    custom_df[time_col] = pd.to_datetime(custom_df[time_col])
    freq_string, periods_per_year = options[sampling_period]
    custom_df = custom_df.set_index(time_col).resample(freq_string).sum()
    with st.expander("Show Resampled Data"):
        st.write("Number of samples:", len(custom_df))
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