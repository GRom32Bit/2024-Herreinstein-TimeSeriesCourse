import numpy as np
import pandas as pd

# for visualization
import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import matplotlib.pyplot as plt
import plotly.graph_objs as go


def plot_ts(ts_set: np.ndarray, plot_title: str = 'Input Time Series Set'):
    ts_num, m = ts_set.shape
    fig = go.Figure()
    for i in range(ts_num):
        fig.add_trace(go.Scatter(x=np.arange(m), y=ts_set[i], line=dict(width=3), name="Time series " + str(i)))
    fig.update_xaxes(showgrid=False,
                     title='Time',
                     title_font=dict(size=18, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=16, color='black'),
                     linewidth=1,
                     tickwidth=1)
    fig.update_yaxes(showgrid=False,
                     title='Values',
                     title_font=dict(size=18, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=16, color='black'),
                     zeroline=False,
                     linewidth=1,
                     tickwidth=1)
    fig.update_layout(title={'text': plot_title, 'x': 0.5, 'y': 0.9, 'xanchor': 'center', 'yanchor': 'top'},
                      title_font=dict(size=18, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      legend=dict(font=dict(size=16, color='black')),
                      width=1800,
                      height=900
                      )
    fig.show()
    plt.show()
