
import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import mysql.connector
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

#CSS Background
app_colors = {
    'background': '#000000',
    'text': '#FFFFFF',
    'sentiment-plot':'#41EAD4',
    'volume-bar':'#FBFC74',
    'someothercolor':'#FF206E',
}

colors = {
    'background': '#000000',
    'text': '#7FDBFF'
}

#Memanggil data dari MySql
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="nopalganteng",
  database="scrap"
)

#dibawah ini ada 2 query untuk manggil data, ada query untuk hari ini dan query seluruh data, tinggal aktifkan aja mau yang mana
mycursor = mydb.cursor()
#data hari ini
#mycursor.execute("SELECT title, date, url, content, author, tag, category FROM new Where date = curdate()")

#alldata
mycursor.execute("SELECT title, date, url, content, author, tag, category, label FROM scrap")
myresult = mycursor.fetchall()
df = pd.DataFrame(myresult)
df.columns = ['Title', 'Date', 'URL', 'Content', 'Author', 'Tag', 'Category','Sentiment']

#PANGGIL DATA UNTUK LINE CHART
#dibawah ini ada 2 query untuk manggil data, ada query untuk hari ini dan query seluruh data, tinggal aktifkan aja mau yang mana
mysen = mydb.cursor()
#data hari ini
#mysen.execute("SELECT date, label FROM new Where date = curdate()")
#alldata
mysen.execute("SELECT sentiment, label FROM scrap")
mysen = mysen.fetchall()
dfsen = pd.DataFrame(mysen)
dfsen.columns = ['sentiment', 'label']

#PANGGIL DATA UNTUK PIE CHART
#dibawah ini ada 2 query untuk manggil data, ada query untuk hari ini dan query seluruh data, tinggal aktifkan aja mau yang mana
mylab = mydb.cursor()
#data hari ini
#mylab.execute("SELECT date, sentiment, label FROM new Where date = curdate()")
#alldata
mylab.execute("SELECT encode, label FROM scrap")
mylab = mylab.fetchall()
dflab = pd.DataFrame(mylab)
dflab.columns = ['encode', 'label']

#data ini untuk tabel, biar bisa title dijadikan link
df_drop_link = df.drop(columns='URL')
       
app = dash.Dash()
app.title = 'Live Detik'

#Pembuatan Dashboard
app.layout = html.Div(
    [
    html.Div(
        #Judul Project
        [html.H1('Dashboard Sentiment of Direktorat Jenderal Kekayaan Negara', style={'color':"#FFFFFF"}),
            html.Div(id="output"),
        ],  
            style={'width':'100%'}),
    
    html.Div([
        html.Div([
            #Pembuatan Graph Line Chart
            dcc.Graph(id='linechart', className="six columns", figure={
                'data': [
                    {'y': [1, 2, 3]}],
                    'layout':{'plot_bgcolor': colors['background'], 'paper_bgcolor': 'black'}
                    })
        ]),
        html.Div([
            #Pembuatan Graph Pie Chart
            dcc.Graph(id='piechart', className="six columns", figure={
                'data': [
                    {'y': [1, 2, 3]}],
                    'layout':{'plot_bgcolor': colors['background'], 'paper_bgcolor': 'black'}
                    }),
        ]),
    ], style={'columnCount': 2}),

    html.Div([
        html.Div([
            #Pembuatan Tabel
            html.Table(
                # Header
                [html.Tr([html.Th(col,style = {'width':'300px'}) for col in df.columns])] +
        
                # Body
                [html.Tr([
                    html.Td(df.iloc[i][col],style = {'max-width': '0',
                                                      'overflow': 'hidden',
                                                      'text-overflow': 'ellipsis',
                                                      'white-space': 'nowrap'}) if col != 'URL' else html.Td(html.A(href=df.iloc[i]['URL'], children=df.iloc[i][col], target='_blank')) for col in df.columns 
                ]) for i in range(len(df))],
                style = {'background': '#000000',
                         'color':'white',
                         'width': '100%',
                         'border': '1px solid white',
                         'font-size':'20px'}
            ),
            #Pembuatan Data tabel agar bisa di panggil di callback
            dash_table.DataTable(
                id='datatable',
                selected_rows=[]
            )
        ])
    ]),
    
    #Interval load page setiap 1 menit
    html.Div([dcc.Interval(
                id='my_interval',
                disabled=False,     #if True, the counter will no longer update
                interval=1*60000,    #increment the counter n_intervals every interval milliseconds
                n_intervals=1,      #number of times the interval has passed
                #max_intervals=4,    #number of times the interval will be fired.
                                    #if -1, then the interval has no limit (the default)
                                    #and if 0 then the interval stops running.
    ),
    html.Div(id='output_data', style={'font-size':30, 'color':"#FFFFFF"}, ),
    dcc.Graph(id="mybarchart", figure={
                'data': [
                    {'y': [1, 2, 3]}],
                    'layout':{'plot_bgcolor': colors['background'], 'paper_bgcolor': 'black'}
                    }), 
    ])
        
    ], style={'backgroundColor': app_colors['background']}, 
    
)


#Callback PieChart dan LineChart
@app.callback(
    [Output('piechart', 'figure'),
     Output('linechart', 'figure')],
    [Input('datatable', 'selected_rows')]
)

def graph_chart(chosen_rows):
    #PieChart
    count = dflab.groupby(['label']).count() 
    pie_chart=px.pie(
        data_frame=count,
        title='TOTAL LABEL SENTIMENT',
        values='encode', 
        labels='label',
        names= ['Negatif','Netral'],
        hole=.3,
        template='plotly_dark').update_layout(
                                   {'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    pie_chart.update_traces(textposition='inside', textinfo='percent+label')
    
    #LineChart
    line_chart = px.line(
        data_frame=dfsen,
        title='NILAI SENTIMENT BERITA',
        y="label",
        hover_data=['sentiment'],
        template='plotly_dark').update_layout(
                                   {'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                                    'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    line_chart.update_layout(uirevision='foo')
    return (pie_chart,line_chart)

#Load Page every 1 mnt
@app.callback(
    [Output('output_data', 'children'),
     Output('mybarchart', 'figure')],
    [Input('my_interval', 'n_intervals')]
)

def loadpage(num):
    if num==0:
        raise PreventUpdate
    else:
        y_data=num
        fig=go.Figure(data=[go.Bar(x=[1], y=[y_data]*1)],
                      layout=go.Layout(yaxis=dict(tickfont=dict(size=10)))
        )
        fig.update_layout(
            showlegend=False,
            template='plotly_dark',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            margin=dict(t=10,l=10,b=10,r=10), 
            height=10, 
            width=2000,
        )
    return (y_data,fig)

if __name__ == '__main__':
    app.run_server()