
import plotly.graph_objects as go
import numpy as np


def dict_barplot(data_dict, max_bars=10):
    """
    util to plot a barplot given dictionary of categorical data.
    Used to report dict metrics to wandb.
    """
    sorted_data = dict(sorted(data_dict.items(), key=lambda item: item[1], reverse=True)[:max_bars])

    categories = list(sorted_data.keys())
    values = np.array(list(sorted_data.values()), dtype=np.int32)

    fig = go.Figure(data=[go.Bar(x=categories, y=values, text=values, textposition='auto')])

    fig.update_layout(font=dict(size=10),
        xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
        margin=dict(l=40, r=40, t=80, b=80),  # Add margin around the plot
    )
    return fig
