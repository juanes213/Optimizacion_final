from flask import Flask, render_template, request, jsonify
import pulp
import pandas as pd
import numpy as np
from pprint import pprint
import plotly.express as px
import plotly
import plotly.io as pio
import json
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
import numpy as np

import warnings 
warnings.filterwarnings('ignore')

app = Flask(__name__)

NPV_por_sectores = {
    'etfs': {'VOO': 215.96818299942478, 'SPXS': 4.662220129633556},
    'salud': {'PFE': 14.603390497452384, 'JNJ': 69.38438717218712, 'GEHC': 33.473227581566455},
    'financiero': {'JPM': 91.1994422394139, 'C': 29.880575226069826, 'BAC': 18.00943089790455},
    'publico': {'ETB': 6.398238318405812},
    'tecnologico': {'GE': 70.26104605034068, 'AAPL': 82.34059622821174, 'MSFT': 186.96096551701095},
    'industrial': {'F': 6.220578290518018}
}

NPV = {
    'VOO': 215.96818299942478, 
    'SPXS': 4.662220129633556, 
    'PFE': 14.603390497452384, 
    'JNJ': 69.38438717218712, 
    'GEHC': 33.473227581566455, 
    'JPM': 91.1994422394139, 
    'C': 29.880575226069826, 
    'BAC': 18.00943089790455,
    'ETB': 6.398238318405812,
    'GE': 70.26104605034068,
    'AAPL': 82.34059622821174,
    'F': 6.220578290518018,
    'MSFT': 186.96096551701095
}

cash_outflow = {
    'VOO': 0.0, 'SPXS': 0.0, 'PFE': 836881000000.0, 'JNJ': 717034000000.0, 
    'GEHC': 87196000000.0, 'JPM': 4638931000000.0, 'C': 3455897000000.0, 
    'BAC': 4351237000000.0, 'ETB': 0.0, 'GE': 647265000000.0, 
    'AAPL': 2722015000000.0, 'F': 902774000000.0, 'MSFT': 1767369000000.0
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model', methods=['GET', 'POST'])
def model():
    if request.method == 'POST':
        sector = request.form['sector']
        num_stocks = int(request.form['num_stocks'])
        max_stocks = int(request.form['max_stocks'])
        max_cash_outflow = float(request.form['max_cash_outflow'])
        implicacion = request.form['implicacion']
        exclusion = request.form['exclusion']

        # Selección de acciones por sector
        escogidos = {}
        all_stocks = list(NPV.keys())
        if sector in NPV_por_sectores:
            for stock in NPV_por_sectores[sector].keys():
                if stock in all_stocks:
                    escogidos[stock] = (NPV[stock], cash_outflow[stock])

        # Formulación del problema de programación lineal
        prob = pulp.LpProblem("Capital_Budgeting_IP", pulp.LpMaximize)
        obj = []

        for stock_escogido in escogidos:
            objeto = pulp.LpVariable(f'{stock_escogido}', cat='Binary')
            obj.append(objeto)

        funcion_objetivo = sum(stock[1][0] * objeto for objeto, stock in zip(obj, escogidos.items()))
        prob += funcion_objetivo

        # Agregar restricción del número máximo de stocks a seleccionar
        prob += sum(obj) <= max_stocks

        primero, segundo = implicacion.split(',')
        prob += obj[int(primero)-1] <= obj[int(segundo)-1]

        primero, segundo = exclusion.split(',')
        prob += obj[int(primero)-1] + obj[int(segundo)-1] <= 1

        prob += sum(stock[1][1] * objeto for objeto, stock in zip(obj, escogidos.items())) <= max_cash_outflow, "Cash outflow"
        prob.solve()

        # Resultados
        resultados = {v.name: v.varValue for v in prob.variables()}
        sensibilidad = {
            "Costes Reducidos": {v.name: v.dj for v in prob.variables()},
            "Precios Sombra": {name: constraint.pi for name, constraint in prob.constraints.items()},
            "Slack": {name: constraint.slack for name, constraint in prob.constraints.items()}
        }

        # Datos para graficar
        costes_r = pd.DataFrame({'name': list(sensibilidad["Costes Reducidos"].keys()), 'rcost': list(sensibilidad["Costes Reducidos"].values())})
        nombres_restricciones = list(sensibilidad["Precios Sombra"].keys())
        precios_sombra = list(sensibilidad["Precios Sombra"].values())
        slack = list(sensibilidad["Slack"].values())

        # Gráfico de Costes Reducidos
        fig1 = px.bar(costes_r, x='name', y='rcost', labels={'x':'Variables', 'y':'Costes Reducidos'}, title='Análisis de Sensibilidad - Costes Reducidos')
        fig1.update_layout(template = 'plotly_dark')
        # Gráfico de Precios Sombra
        fig2 = px.bar(x=nombres_restricciones, y=precios_sombra, labels={'x':'Restricciones', 'y':'Precios Sombra'}, title='Análisis de Sensibilidad - Precios Sombra')
        fig2.update_layout(template = 'plotly_dark')
        # Gráfico de Slack
        fig3 = px.bar(x=nombres_restricciones, y=slack, labels={'x':'Restricciones', 'y':'Slack'}, title='Análisis de Sensibilidad - Slack')
        fig3.update_layout(template = 'plotly_dark')
        # Convertir las figuras a JSON para pasarlas al template
        fig1_json = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
        fig2_json = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
        fig3_json = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

        # Generar HTML para resultados
        results_html = f"<h2>Resultados</h2><ul>"
        for key, value in resultados.items():
            results_html += f"<li>{key}: {value}</li>"
        results_html += "</ul>"

        return jsonify({
            'results_html': results_html,
            'fig1': json.loads(fig1_json),
            'fig2': json.loads(fig2_json),
            'fig3': json.loads(fig3_json)
        })

    return render_template('model.html')


@app.route('/visualization')
def visualization():
    # Bar Plot del NPV por Sectores
    sectores = list(NPV_por_sectores.keys())
    npv_sectores = [sum(NPV_por_sectores[sector].values()) for sector in sectores]

    fig1 = px.bar(x=sectores, y=npv_sectores, labels={'x': 'Sectores', 'y': 'NPV Total'}, title='NPV Total por Sectores')
    fig1.update_layout(template = 'plotly_dark' )
    # Pie Chart del NPV Total por Sectores
    fig2 = px.pie(values=npv_sectores, names=sectores, title='Proporción del NPV Total por Sectores', hole=0.3)
    fig2.update_layout(template = 'plotly_dark' )
    # Bar Plot de NPV Individual
    acciones = list(NPV.keys())
    npv_acciones = list(NPV.values())

    fig3 = px.bar(x=acciones, y=npv_acciones, labels={'x': 'Acciones', 'y': 'NPV'}, title='NPV de cada Acción')
    fig3.update_layout(xaxis_tickangle=-45, template = 'plotly_dark')

    # Convertir las figuras a JSON para pasarlas al template
    fig1_json = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    fig2_json = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    fig3_json = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('visualization.html', fig1_json=fig1_json, fig2_json=fig2_json, fig3_json=fig3_json)

if __name__ == "__main__":
    app.run(debug=True)
