import plotly.graph_objs as go
from pandas import read_excel
from plotly.offline import plot

from dashboard.logic.machine_learning.predict import predict
from dashboard.logic.machine_learning.settings import scale_name, prediction_name, hilev_prediction
from dashboard.models import SleepNight, CsvData


def create_graph(d):
    if isinstance(d, CsvData):
        df = predict(d)
        sleep_predicted = _sleep_data_from_data_frame(df,
                                                      prediction_name,
                                                      'Predicted data',
                                                      '#8dcb89')
        if d.training_data:
            sleep_ps = _sleep_data_from_data_frame(df,
                                                   scale_name,
                                                   'Polysomnography data',
                                                   '#6ec06a')
            fig = go.Figure(
                data=[
                    sleep_ps,
                    sleep_predicted
                ]
            )
        else:
            fig = go.Figure(
                data=[
                    sleep_predicted
                ]
            )
        title = \
            f'Body location: {d.get_body_location_display()} | Creation date: {d.creation_date}' \
                if not d.description else \
                f'Body location: {d.get_body_location_display()} | Creation date: {d.creation_date} | ' \
                f'Description: {d.description}'
    elif isinstance(d, SleepNight):
        sleep_predicted = _sleep_data_from_sleep_night(d, hilev_prediction, 'Sleep prediction', '#8dcb89')
        fig = go.Figure(
            data=[
                sleep_predicted
            ]
        )
        title = \
            f'Body location: {d.data.get_body_location_display()} | Creation date: {d.data.creation_date}' \
                if not d.data.description else \
                f'Body location: {d.data.get_body_location_display()} | Creation date: {d.data.creation_date} | ' \
                f'Description: {d.data.description}'
    else:
        return

    fig.update_layout(
        title_text=title,
        plot_bgcolor='rgba(240,240,240,240)',
        barmode='group',
        yaxis={'categoryorder': 'category descending'}
    )
    fig.update_xaxes(rangeslider_visible=True,
                     rangeselector=dict(
                         buttons=list([
                             dict(count=1, label="1m", step="minute", stepmode="backward"),
                             dict(count=5, label="5m", step="minute", stepmode="backward"),
                             dict(count=10, label="10m", step="minute", stepmode="backward"),
                             dict(count=30, label="30m", step="minute", stepmode="backward"),
                             dict(count=1, label="1h", step="hour", stepmode="backward"),
                             dict(step="all")
                         ])
                     )
                     )
    plot_div = plot(figure_or_data=fig, output_type='div')
    return plot_div


def _sleep_data_from_data_frame(df, column, name, color):
    x = df.index
    map_values = {True: 'Sleep', False: 'Wake'}
    y = df[column].astype(bool).map(map_values)
    sleep_ps = go.Bar(x=x, y=y, name=name, marker_color=color)
    return sleep_ps


def _sleep_data_from_sleep_night(sleep_night, column, name, color):
    df = read_excel(sleep_night.name, index_col=0)
    x = df.index
    map_values = {'S': 'Sleep', 'W': 'Wake'}
    y = df[column].astype(str).map(map_values)
    sleep_ps = go.Bar(x=x, y=y, name=name, marker_color=color)
    return sleep_ps
