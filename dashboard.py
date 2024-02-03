import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Wczytywanie modelu
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Utworzenie aplikacji Dash
app = dash.Dash(__name__)

# Layout aplikacji
app.layout = html.Div([
    # Nagłówek
    html.H1('Analiza Regresji Liniowej', style={'textAlign': 'center'}),
    
    # Sekcja wyboru danych
    html.Div([
        # Dropdown do wyboru zestawu danych
        html.Div([
            html.H3('Wybierz zestaw danych:'),
            dcc.Dropdown(
                id='data-set-dropdown',
                options=[
                    {'label': 'Dane treningowe', 'value': 'train'},
                    {'label': 'Dane testowe', 'value': 'test'}
                ],
                value='test'
            )
        ]),
        
        # Dropdown do wyboru cechy
        html.Div([
            html.H3('Wybierz cechę:'),
            dcc.Dropdown(
                id='feature-dropdown',
                options=[],
                value=None
            )
        ]),
        
        # Dropdown do wyboru typu wykresu
        html.Div([
            html.H3('Wybierz typ wykresu:'),
            dcc.Dropdown(
                id='chart-type-dropdown',
                options=[
                    {'label': 'Wykres regresji', 'value': 'regression'},
                    {'label': 'Histogram', 'value': 'histogram'},
                    {'label': 'Boxplot', 'value': 'boxplot'}
                ],
                value='regression'
            )
        ])
    ]),
    
    # Wykres
    dcc.Graph(id='chart'),
    
    # Sekcja metryk
    html.Div(id='metrics'),
    
    # Sekcja losowej próbki danych
    html.Div([
        html.H3('Losowa próbka danych'),
        html.Div(id='data-table')
    ])
])

# Callback do aktualizacji opcji dropdown dla cech w zależności od wybranego zestawu danych
@app.callback(
    [Output('feature-dropdown', 'options'),
     Output('feature-dropdown', 'value')],
    [Input('data-set-dropdown', 'value')]
)
def update_feature_dropdown(selected_set):
    if selected_set == 'train':
        df = pd.read_csv('xtrain_selected.csv').round(2)
    else:
        df = pd.read_csv('xtest_selected.csv').round(2)

    features = [{'label': col, 'value': col} for col in df.columns if col != 'price']
    return features, features[0]['value']

# Callback dla wykresu, metryk i tabeli danych
@app.callback(
    [Output('chart', 'figure'),
     Output('metrics', 'children'),
     Output('data-table', 'children')],
    [Input('data-set-dropdown', 'value'),
     Input('feature-dropdown', 'value'),
     Input('chart-type-dropdown', 'value')]
)

def update_graph_metrics_table(selected_set, selected_feature, chart_type):
    if selected_set == 'train':
        df = pd.read_csv('xtrain_selected.csv')
    else:
        df = pd.read_csv('xtest_selected.csv')

    X = df.drop('price', axis=1)
    y = df['price']
    y_pred = model.predict(X)

    # Tworzenie wykresu w zależności od wybranego typu
    if chart_type == 'regression':
        X_feature = pd.DataFrame(np.tile(X.mean().values, (df.shape[0], 1)), columns=X.columns)
        X_feature[selected_feature] = df[selected_feature].values
        y_pred_line = model.predict(X_feature)

        figure = {
            'data': [
                go.Scatter(
                    x=df[selected_feature],
                    y=y,
                    mode='markers',
                    marker={'size': 15},
                    name='Punkty danych'
                ),
                go.Scatter(
                    x=df[selected_feature],
                    y=y_pred_line,
                    mode='lines',
                    name='Linia regresji'
                )
            ],
            'layout': go.Layout(
                title=f'Cena vs {selected_feature}',
                xaxis={'title': selected_feature},
                yaxis={'title': 'Cena'}
            )
        }
    elif chart_type == 'histogram':
        figure = {
            'data': [
                go.Histogram(
                    x=df[selected_feature]
                )
            ],
            'layout': go.Layout(
                title=f'Histogram cechy {selected_feature}',
                xaxis={'title': selected_feature},
                yaxis={'title': 'Liczba obserwacji'}
            )
        }
    elif chart_type == 'boxplot':
        figure = {
            'data': [
                go.Box(
                    y=df[selected_feature]
                )
            ],
            'layout': go.Layout(
                title=f'Boxplot cechy {selected_feature}',
                xaxis={'title': selected_feature},
                yaxis={'title': 'Wartości'}
            )
        }

    rmse = mean_squared_error(y, y_pred, squared=False)
    r2 = r2_score(y, y_pred)
    metrics = html.Div([
        html.H3('Błąd RMSE oraz współczynnik R2'),
        html.P(f'RMSE: {rmse : .2f}'),
        html.P(f'R2: {r2 : .2f}')
    ])

    # Dodanie przewidywań i rzeczywistych wartości do DataFrame
    sample_df = df.sample(5).round(2)
    sample_df['Predykcja'] = model.predict(sample_df.drop('price', axis=1)).round(2)
    
    data_table = html.Table(
        [html.Tr([html.Th(col) for col in sample_df.columns])] +
        [html.Tr([html.Td(sample_df.iloc[i][col]) for col in sample_df.columns]) for i in range(len(sample_df))]
    )

    return figure, metrics, data_table

# Uruchomienie aplikacji
if __name__ == '__main__':
    app.run_server(debug=True)
