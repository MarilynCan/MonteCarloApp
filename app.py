import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from loss_models import PoissonFrequency, LognormalMagnitude, SimpleLoss, MultiLoss, Loss
import pandas as pd
import altair as alt

def simulate_losses(event_data, num_years):
    losses = []
    df_loss_summary = pd.DataFrame(index=['Min', 'Moda', 'Mediana', 'Max', 'Media', 'P10', 'P20', 'P30', 'P40', 'P50', 'P60', 'P70', 'P80', 'P90', 'P95', 'P98', 'P99', 'P99.5', 'P99.9', 'P99.99'])
    for i, event in enumerate(event_data):
        freq = event['frecuencia']
        low_loss = event['low_loss']
        high_loss = event['high_loss']
        loss_model = SimpleLoss(label=f'Evento {i + 1}',
                                name=f'Evento {i + 1}',
                                frequency=freq,
                                low_loss=low_loss,
                                high_loss=high_loss)
        simulated_losses_one = loss_model.simulate_years(num_years)
        loss_summary_one = loss_model.summarize_loss(np.array(simulated_losses_one))
        losses.append(loss_model)
        df_loss_summary[f'Evento {i + 1}'] = pd.Series(loss_summary_one)  # Agregar la columna al DataFrame

    multi_loss = MultiLoss(losses)
    simulated_losses = multi_loss.simulate_years(num_years)
    Cp_mean = np.sum(np.array(simulated_losses)) / num_years
    df_loss_summary = df_loss_summary.style.format(formatter="{:,.1f}")
    prioritized_losses = multi_loss.prioritized_losses()
    prioritized_losses_df = pd.DataFrame(prioritized_losses, columns=['Nombre Evento', 'Nombre', 'Ciberperdida Media Anual'])

    return Cp_mean,prioritized_losses_df, df_loss_summary, simulated_losses

# Función para graficar el scatter plot
def plot_scatter(losses):
    years = range(1, len(losses) + 1)
    data = pd.DataFrame({'Years': years, 'Losses': losses})
    chart = alt.Chart(data).mark_circle().encode(
        x='Years',
        y='Losses',
        tooltip=['Years', 'Losses']
    ).properties(
        width=800,
        height=500
    )
    st.altair_chart(chart, use_container_width=True)

def find_first_positive(arr):
    for num in arr:
        if num > 0:
            return num
    return None  # Si no se encontró ningún valor mayor que cero


def loss_exceedance_curve(simulated_losses):
    # Calcula los valores de pérdida para diferentes percentiles
    losses = np.array([np.percentile(list(simulated_losses), x) for x in range(1, 100, 1)])
    percentiles = np.array([float(100 - x) / 100.0 for x in range(1, 100, 1)])

    # Calcula los límites mínimos y máximos de las pérdidas simuladas
    x_min = find_first_positive(losses)
    x_max = max(losses)

    # Crea un DataFrame para los datos de pérdida y percentiles
    data_df = pd.DataFrame({'Losses': losses, 'Percentiles': percentiles})

    # Crea el gráfico Altair
    chart = alt.Chart(data_df).transform_filter(alt.datum.Losses > 0).mark_line().encode(
        x=alt.X('Losses', scale=alt.Scale(type="log", domain=[x_min, x_max]),  axis=alt.Axis(format=',.2r')),
        y=alt.Y('Percentiles', axis=alt.Axis(format='.0%'))
    )

    # Configura los límites del eje y para que siempre muestre el rango completo del 0% al 100%
    #chart = chart.encode(y=alt.Y('Percentiles', axis=alt.Axis(format='%'), scale=alt.Scale(domain=[0, 1.0])))

    # Muestra las líneas de cuadrícula en ambos ejes
    chart = chart.configure_axis(grid=True)
    st.altair_chart(chart, use_container_width=True)

def main():
    st.title('Simulación Monte Carlo')

    # Usamos 'with' para crear una columna a la izquierda
    with st.sidebar:
        num_years = st.number_input('Ingrese el número de años a simular', min_value=1, value=1000)
        num_events = st.number_input('Ingrese la cantidad de eventos a simular', min_value=1, value=1)

        event_data = []
        for i in range(num_events):
            st.write(f'Evento {i + 1}')
            freq = st.number_input(f'Frecuencia del evento {i + 1}', min_value=0.01,max_value=1.0, value=0.5)
            low_loss = st.number_input(f'Pérdida mínima del evento {i + 1}', value=0)
            high_loss = st.number_input(f'Pérdida máxima del evento {i + 1}', value=0)
            event_data.append({'frecuencia': freq, 'low_loss': low_loss, 'high_loss': high_loss})

    # Verificar si se hizo clic en el botón "Simular"
    if st.sidebar.button('Simular'):
        Cp_mean ,prioritized_losses_df, df_loss_summary , losses = simulate_losses(event_data, num_years)
        # Primer gráfico
        st.subheader("Ciberperdida Media Anual Agregada:")
        # Centrar y remarcar el título Cp_mean
        #st.markdown(f'<div style="text-align:center;font-size:1.5em;font-weight:bold;">Cp_mean</div>', unsafe_allow_html=True)
        # Centrar el valor numérico de Cp_mean
        st.markdown(f'<div style="text-align:center;font-size:2em;background-color:#fce4ec;padding:5px;">{Cp_mean:,.1f}</div>', unsafe_allow_html=True)

        # Segungo gráfico
        st.subheader("Simulación de pérdidas")
        plot_scatter(losses)

        #  Tercer gráfico
        st.subheader("Curva de Excedencia de Pérdidas")
        loss_exceedance_curve(losses)

        # Cuarto gráfico
        st.subheader("Resumen de las pérdidas:")
        st.table(df_loss_summary)

        # Quinto gráfico
        st.subheader("Pérdidas priorizadas:")
        st.table(prioritized_losses_df[['Nombre', 'Ciberperdida Media Anual']])

if __name__ == '__main__':
    st.set_option('deprecation.showPyplotGlobalUse', False)  # Deshabilitar la advertencia
    main()