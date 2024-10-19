
import plotly.graph_objects as go
from melee import Action
import numpy as np


def dict_barplot(data_dict, color='skyblue', max_bars=10):
    sorted_data = dict(sorted(data_dict.items(), key=lambda item: item[1], reverse=True)[:max_bars])

    categories = list(sorted_data.keys())
    values = np.array(list(sorted_data.values()), dtype=np.int32)

    fig = go.Figure(data=[go.Bar(x=categories, y=values, text=values, textposition='auto')])

    fig.update_layout(font=dict(size=10),
        #plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        #paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
        xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
        margin=dict(l=40, r=40, t=80, b=80),  # Add margin around the plot
    )
    return fig

if __name__ == '__main__':

    d = {
        a.name: np.random.randint(0, 10) for a in Action
    }

    dict_barplot(d).show()