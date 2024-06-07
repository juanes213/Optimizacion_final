from flask import Flask, render_template, request, jsonify
import pulp
import pandas as pd
import numpy as np
import plotly.express as px
import plotly
import plotly.io as pio
import json

import warnings 
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Datos de ejemplo para NPV, flujo de efectivo y penalización de riesgo
NPV_por_sectores = {
    'etfs': {'VOO': 215.968, 'SPXS': 4.662},
    'salud': {'PFE': 14.603, 'JNJ': 69.384, 'GEHC': 33.473},
    'financiero': {'JPM': 91.199, 'C': 29.880, 'BAC': 18.009},
    'publico': {'ETB': 6.398},
    'tecnologico': {'GE': 70.261, 'AAPL': 82.340, 'MSFT': 186.960},
    'industrial': {'F': 6.220}
}

NPV = {k: v for sector in NPV_por_sectores.values() for k, v in sector.items()}
cash_outflow = {
    'VOO': 0.0, 'SPXS': 0.0, 'PFE': 836881000000.0, 'JNJ': 717034000000.0, 
    'GEHC': 87196000000.0, 'JPM': 4638931000000.0, 'C': 3455897000000.0, 
    'BAC': 4351237000000.0, 'ETB': 0.0, 'GE': 647265000000.0, 
    'AAPL': 2722015000000.0, 'F': 902774000000.0, 'MSFT': 1767369000000.0
}

# Penalizaciones de riesgo y retorno esperado
risk_penalty = {
    'VOO': 0.1, 'SPXS': 0.3, 'PFE': 0.2, 'JNJ': 0.15, 'GEHC': 0.18,
    'JPM': 0.12, 'C': 0.25, 'BAC': 0.2, 'ETB': 0.3, 'GE': 0.22,
    'AAPL': 0.18, 'F': 0.25, 'MSFT': 0.15
}
expected_return = {
    'VOO': 0.05, 'SPXS': 0.07, 'PFE': 0.06, 'JNJ': 0.055, 'GEHC': 0.065,
    'JPM': 0.06, 'C': 0.05, 'BAC': 0.055, 'ETB': 0.06, 'GE': 0.07,
    'AAPL': 0.065, 'F': 0.05, 'MSFT': 0.075
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
        min_cash_outflow = float(request.form['min_cash_outflow'])
        implicacion = request.form.get('implicacion')
        exclusion = request.form.get('exclusion')

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

        # Función objetivo mejorada
        funcion_objetivo = sum((expected_return[stock] - risk_penalty[stock]) * objeto for stock, objeto in zip(escogidos.keys(), obj))
        prob += funcion_objetivo

        # Agregar restricciones
        prob += sum(obj) <= max_stocks, "Max Stocks"
        prob += sum(stock[1][1] * objeto for objeto, stock in zip(obj, escogidos.items())) <= max_cash_outflow, "Max Cash Outflow"
        prob += sum(stock[1][1] * objeto for objeto, stock in zip(obj, escogidos.items())) >= min_cash_outflow, "Min Cash Outflow"

        # Restricciones de implicación y exclusión
        if implicacion:
            try:
                primero, segundo = map(int, implicacion.split(','))
                if primero <= len(obj) and segundo <= len(obj):
                    prob += obj[primero-1] <= obj[segundo-1]
            except ValueError:
                pass  # Handle improper input gracefully
        if exclusion:
            try:
                primero, segundo = map(int, exclusion.split(','))
                if primero <= len(obj) and segundo <= len(obj):
                    prob += obj[primero-1] + obj[segundo-1] <= 1
            except ValueError:
                pass  # Handle improper input gracefully

        prob.solve()

        estado_optimo = pulp.LpStatus[prob.status]
        es_optimo = prob.status == pulp.LpStatusOptimal

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
        fig1.update_layout(template='plotly_dark')
        # Gráfico de Precios Sombra
        fig2 = px.bar(x=nombres_restricciones, y=precios_sombra, labels={'x':'Restricciones', 'y':'Precios Sombra'}, title='Análisis de Sensibilidad - Precios Sombra')
        fig2.update_layout(template='plotly_dark')
        # Gráfico de Slack
        fig3 = px.bar(x=nombres_restricciones, y=slack, labels={'x':'Restricciones', 'y':'Slack'}, title='Análisis de Sensibilidad - Slack')
        fig3.update_layout(template='plotly_dark')

        # Convertir las figuras a JSON para pasarlas al template
        fig1_json = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
        fig2_json = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
        fig3_json = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

        results_html = f"<h2>Resultados</h2>"
        results_html += f"<p>Estado de la solución: <strong>{estado_optimo}</strong></p>"
        if es_optimo:
            results_html += "<ul>"
            for key, value in resultados.items():
                results_html += f"<li>{key}: {value}</li>"
            results_html += "</ul>"
        else:
            results_html += "<p>No se encontró una solución óptima. Revisa las restricciones o cambia los parámetros.</p>"


        return jsonify({
            'results_html': results_html,
            'fig1': json.loads(fig1_json),
            'fig2': json.loads(fig2_json),
            'fig3': json.loads(fig3_json)
        })

    return render_template('model.html')



@app.route('/visualization')
def visualization():
    # Datos para gráficos
    sectores = list(NPV_por_sectores.keys())
    npv_sectores = [sum(NPV_por_sectores[sector].values()) for sector in sectores]
    acciones = list(NPV.keys())
    npv_acciones = list(NPV.values())

    # Gráfico de NPV Total por Sectores
    fig1 = px.bar(x=sectores, y=npv_sectores, labels={'x': 'Sectores', 'y': 'NPV Total'}, title='NPV Total por Sectores')
    fig1.update_layout(template='plotly_dark')

    # Gráfico de Proporción del NPV Total por Sectores
    fig2 = px.pie(values=npv_sectores, names=sectores, title='Proporción del NPV Total por Sectores', hole=0.3)
    fig2.update_layout(template='plotly_dark')
    # Gráfico de NPV de cada Acción
    fig3 = px.bar(x=acciones, y=npv_acciones, labels={'x': 'Acciones', 'y': 'NPV'}, title='NPV de cada Acción')
    fig3.update_layout(xaxis_tickangle=-45, template='plotly_dark')

    # Tres gráficos adicionales (por ejemplo, distribución de NPV por sector)
    npv_data = [{'Sector': sector, 'Stock': stock, 'NPV': npv} for sector, stocks in NPV_por_sectores.items() for stock, npv in stocks.items()]
    npv_df = pd.DataFrame(npv_data)

    fig4 = px.histogram(npv_df, x='Sector', y='NPV', color='Sector', labels={'NPV': 'NPV'}, title='Distribución del NPV por Sector')
    fig4.update_layout(template='plotly_dark')
    fig5 = px.scatter(npv_df, x='Stock', y='NPV', color='Sector', labels={'NPV': 'NPV'}, title='NPV por Stock')
    fig5.update_layout(template='plotly_dark')
    fig6 = px.box(npv_df, x='Sector', y='NPV', color='Sector', labels={'NPV': 'NPV'}, title='Box Plot del NPV por Sector')
    fig6.update_layout(template='plotly_dark')
    # Convertir las figuras a JSON para pasarlas al template
    fig1_json = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    fig2_json = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    fig3_json = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    fig4_json = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
    fig5_json = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)
    fig6_json = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('visualization.html', fig1_json=fig1_json, fig2_json=fig2_json, fig3_json=fig3_json, fig4_json=fig4_json, fig5_json=fig5_json, fig6_json=fig6_json)


if __name__ == "__main__":
    app.run(debug=True)
