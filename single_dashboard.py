import os

from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import numpy as np
import pandas as pd

from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform, html
import plotly.graph_objects as go

import joblib
import torch
from sklearn.metrics import mean_squared_error

import torch

class AE(torch.nn.Module):
  def __init__(self):
    super(AE, self).__init__()
    self.encoder = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.Tanh(),
            torch.nn.Linear(1024, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 256),
            torch.nn.Tanh(),
        )
    self.decoder = torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 1024),
            torch.nn.Tanh(),
            torch.nn.Linear(1024, 2048),
            torch.nn.Tanh(),
        )
  def forward(self, x):
          encoded = self.encoder(x)
          decoded = self.decoder(encoded)
          return decoded

global f
f = np.linspace(0, 1000, 2048)

def softmax_calculator(input_diction):
  output_diction = {}

  for state in list(input_diction.keys()):
    output_diction[state] = np.exp(input_diction[state])

  summation = np.sum(np.array(list(output_diction.values())))

  normalized_output_diction = {}

  for state in list(output_diction.keys()):
    normalized_output_diction[state] = np.round(output_diction[state]/summation, 4)

  return normalized_output_diction

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = DashProxy(__name__,transforms=[MultiplexerTransform()], external_stylesheets=external_stylesheets)
server = app.server

importing_path = os.getcwd() + '/assets/'
sig_df = pd.read_csv(importing_path + 'subsampled_test_df.csv', index_col=0)
lrp_df = pd.read_csv(importing_path + 'LRP_subsampled_test_df.csv', index_col=0)
feature_scaler = joblib.load(open(importing_path + 'scaler.pkl', 'rb'))
AE_loaded = AE()
AE_loaded.load_state_dict(torch.load(importing_path + 'ae.pt', map_location=torch.device('cpu')))
AE_loaded.eval()




app.layout = html.Div(children=[
    html.Div(children = [
        dcc.Graph(
        id='singal-lrp-graph',
        style={'height':'750'}
        ),

        html.Br(),


        html.Div(children = [

            html.Div(children=[
                html.Label('Annotation Rotational Speed (RPM)', style={'text-align':'center'}),
                dcc.Dropdown(id = 'annotation_rpm', options = ['1730', '1750', '1772', '1797']),
                ], className="three columns"),

            html.Div(children=[
                html.Label('State Annotation', style={'text-align':'center'}),
                dcc.Dropdown(id = 'annotation_state', options = ['Inner-Race', 'Outer-Race', 'Ball']),
                ], className="three columns"),

            html.Div(children=[
                html.Label('Fault Frequency', style={'text-align':'center'}),
                html.Div(id = 'fault_freq', style={'text-align':'center'}),
                ], className="three columns"),

            html.Div(children=[
                html.Label('Normalized Similarity Score', style={'text-align':'center'}),
                html.Div(id = 'normalized_similarity_score', style={'text-align':'center'}),
                ], className="three columns"),

        ], className="row", style={'text-align':'center'})
            

    ],className="nine columns"),
    html.Div(children=[
        html.H4('Select Signal Properties', style={'text-align':'center'}),

        html.Br(),

        html.Div(children=[
            html.Label('Rotational Speed (RPM)', style={'text-align':'center'}),
            dcc.Dropdown(id = 'rpm', options = ['1730', '1750', '1772', '1797']),
            ]),
        
        html.Br(),

        html.Div(children=[
            html.Label('Health State', style={'text-align':'center'}),
            dcc.Dropdown(id = 'state', options = ['Normal', 'InnerRace', 'OuterRace', 'Ball'],),
            ]),

        html.Br(),

        html.Div(children=[
            html.Label('Repeatation', style={'text-align':'center'}),
            dcc.Dropdown(id = 'rep', options = [str(rep) for rep in range(1,21)],),
            ]),

        html.Br(),

        html.Div(children=[
            html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
        ], style={'text-align':'center'}),


        html.Div(children=[
            html.H5(id = 'health_index', style={'text-align':'center'}),
            ]),

        html.Br(),

        html.Div(children=[
            html.H6('Bearing Specifications!', style={'text-align':'center'}),
            html.Div(
                html.A(
                    children = html.Img(src = r'assets/qr-code.png', alt='image', style={'height':'50%', 'width':'50%', }),
                    href = 'https://www.skf.com/group/products/rolling-bearings/ball-bearings/deep-groove-ball-bearings/productid-6205',
                    target="_blank",
                ),
                style = {'text-align':'center'}
                )
            ]),
        
        ], className="three columns"),
], className="row")

fault_freq_ratios = {
    'Inner-Race': 5.4152,
    'Ball': 4.7135,
    'Outer-Race': 3.5848,
}

rpm_state_ratio_dict = {
    '1730':
    {
        'Inner-Race': 156.14,
        'Outer-Race': 103.36,
        'Ball': 135.91
    },
    '1750':
    {
        'Inner-Race': 157.94,
        'Outer-Race': 104.56,
        'Ball': 137.48
    },
    '1772':
    {
        'Inner-Race': 159.93,
        'Outer-Race': 105.87,
        'Ball': 139.21
    },
    '1797':
    {
        'Inner-Race': 162.19,
        'Outer-Race': 107.36,
        'Ball': 141.17
    }
}



@app.callback(
    Output(component_id='health_index', component_property='children'),
    Output(component_id='health_index', component_property='style'),
    Output(component_id='singal-lrp-graph', component_property='figure'),
    Input('submit-button-state', 'n_clicks'),
    State(component_id='rpm', component_property='value'),
    State(component_id='state', component_property='value'),
    State(component_id='rep', component_property='value'),
)
def select_draw_signal(n_clicks,rpm,state,rep):

    if rpm and state and rep:
        selected_sig = sig_df.loc[(sig_df['state'] == state) & (sig_df['load'] == int(rpm))].iloc[int(rep), :]
        selected_lrp = lrp_df.loc[(lrp_df['state'] == state) & (lrp_df['load'] == int(rpm))].iloc[int(rep), :]
        
        global x
        x = selected_sig[:2048].to_numpy()

        global lrp
        lrp = selected_lrp[:2048].to_numpy()

        x_scaled = feature_scaler.transform(x.reshape(1,-1))
        x_scaled_VAR = torch.autograd.Variable(torch.Tensor(x_scaled).float())
        x_recons =  AE_loaded(x_scaled_VAR).cpu().detach().numpy()

        mse = np.round(mean_squared_error(x_scaled, x_recons), 4)

        health_index = 'Health State Indicator: ' + str(mse)
        if mse < 0.5:
            color = 'green'
        else:
            color = 'red'

        style = {
            'color': color,
            'text-align':'center'
        }

        global f
        f = np.linspace(0, 1000, 2048)
        fig_signal = px.line(x = list(f), y = [list(x), list(lrp)],)
        series_name = ['Signal', 'LRP']

        for idx, name in enumerate(series_name):
            fig_signal.data[idx].name = name
            # fig_signal.data[idx].hovertemplate = name
        
        fig_signal.update_layout(title='Original Signal & LRP',
                                xaxis_title='Hz',
                                yaxis_title='Amplitude',
                                title_x=0.5,
                                legend_title_text='Variable')

    
    return health_index, style, fig_signal

@app.callback(
    Output(component_id='singal-lrp-graph', component_property='figure'),
    Output(component_id='fault_freq', component_property='children'),
    Output(component_id='normalized_similarity_score', component_property='children'),
    Input(component_id='rpm', component_property='value'),
    Input(component_id='state', component_property='value'),
    Input(component_id='rep', component_property='value'),
    Input(component_id='annotation_rpm', component_property='value'),
    Input(component_id='annotation_state', component_property='value'),
    Input(component_id='singal-lrp-graph', component_property='figure'),
)
def annotation_updater(rpm,state,rep,annotation_rpm, annotation_state, fig_signal):
    if annotation_rpm and annotation_state and fig_signal:

        annotation_freq = rpm_state_ratio_dict[annotation_rpm][annotation_state]
        fig_signal = go.Figure(fig_signal)
        harmonics = 3
        harmonic_severity_ratio = [1, 0.66, 0.33]
        similarity_annotation_vectors = {}
        similarity_scores = {}

        for i in range(1, harmonics + 1):
            fig_signal.add_vline(x = i*annotation_freq, line_dash = 'dash', annotation_text = str(i) + 'X' + annotation_state)

        for state in list(fault_freq_ratios.keys()):
            ratio = fault_freq_ratios[state]
            temp_annotation = np.zeros(f.shape)
            for i in range(1, harmonics+1):
                freq_range = [i * int(annotation_rpm)/60 * ratio - 10, i * int(annotation_rpm)/60 * ratio + 10]
                temp_annotation[np.where(np.logical_and(f >= freq_range[0], f <= freq_range[1]))] = harmonic_severity_ratio[i - 1]
                similarity_annotation_vectors[state] = temp_annotation

            similarity_scores[state] = np.inner(x, similarity_annotation_vectors[state])


        normalized_similarity_scores = softmax_calculator(similarity_scores)
        normalized_similarity_score = normalized_similarity_scores[annotation_state]

        return fig_signal, str(annotation_freq) + ' Hz', normalized_similarity_score

    elif (annotation_rpm is None or annotation_state is None) and (rpm and state and rep):

        fig_signal = px.line(x = list(f), y = [list(x), list(lrp)],)
        series_name = ['Signal', 'LRP']

        for idx, name in enumerate(series_name):
            fig_signal.data[idx].name = name
            # fig_signal.data[idx].hovertemplate = name
        
        fig_signal.update_layout(title='Original Signal & LRP',
                                xaxis_title='Hz',
                                yaxis_title='Amplitude',
                                title_x=0.5,
                                legend_title_text='Variable')

        return fig_signal, '', ''

if __name__ == '__main__':
    app.run_server(debug=True)