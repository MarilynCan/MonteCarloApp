import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import PoissonFrequency, LognormalMagnitude, SimpleLoss, MultiLoss, Loss
import pandas as pd
import altair as alt
from PIL import Image

img = Image.open('images/logo.png')
st.set_page_config(page_title='Alicorp', page_icon=img)


def simulate_losses(event_data, time_unit):
    """
    Simulate losses based on event data and the number of time units.

    Parameters:
        event_data (list): A list of dictionaries containing information about each event.
                           Each dictionary must have keys 'frecuencia', 'low_loss', and 'high_loss'.
        num_years (int): The number of time units to simulate.

    Returns:
        Cp_mean: The mean cyber loss (Cp_mean) calculated from the simulated losses.
        prioritized_losses_df: DataFrame containing prioritized losses with columns 'Nombre Evento', 'Nombre', and 'Ciberperdida Media'.
        df_loss_summary: DataFrame containing the summary statistics of the simulated losses.
        simulated_losses: Array containing the simulated losses.
    """
    losses = []
    # Create an empty DataFrame to store the summary statistics of the simulated losses
    df_loss_summary = pd.DataFrame(index=['Min', 'Moda', 'Mediana', 'Max', 'Media', 'P10', 'P20', 'P30', 'P40', 'P50', 'P60', 'P70', 'P80', 'P90', 'P95', 'P98', 'P99', 'P99.5', 'P99.9', 'P99.99'])

    # Loop through each event and simulate losses
    for i, event in enumerate(event_data):
        freq = event['frecuencia']
        low_loss = event['low_loss']
        high_loss = event['high_loss']
        
        # Create a SimpleLoss model for the current event
        loss_model = SimpleLoss(label=f'Evento {i + 1}',
                                name=f'Evento {i + 1}',
                                frequency=freq,
                                low_loss=low_loss,
                                high_loss=high_loss)
        
        # Simulate losses for the current event over the given number of time units
        simulated_losses_one = loss_model.simulate_years(time_unit)
        
        # Calculate and store the summary statistics for the current event losses
        loss_summary_one = loss_model.summarize_loss(np.array(simulated_losses_one))
        losses.append(loss_model)
        df_loss_summary[f'Evento {i + 1}'] = pd.Series(loss_summary_one)  # Add the column to the DataFrame

    # Create a MultiLoss model to aggregate losses from all events
    multi_loss = MultiLoss(losses)

    # Simulate aggregated losses over the given number of time units
    simulated_losses = multi_loss.simulate_years(time_unit)

    # Calculate the mean cyber loss (Cp_mean) from the simulated losses
    Cp_mean = np.sum(np.array(simulated_losses)) / time_unit

    # Format the DataFrame to show numbers with one decimal place
    df_loss_summary = df_loss_summary.style.format(formatter="{:,.1f}")

    # Get the prioritized losses as a DataFrame with columns 'Nombre Evento', 'Nombre', and 'Ciberperdida Media'
    prioritized_losses = multi_loss.prioritized_losses()
    prioritized_losses_df = pd.DataFrame(prioritized_losses, columns=['Nombre Evento', 'Nombre', 'Ciberperdida Media'])

    return Cp_mean, prioritized_losses_df, df_loss_summary, simulated_losses


def plot_scatter(losses, time_unit):
    """
    Plot a scatter chart for the given losses over time.

    Parameters:
        losses (list or numpy.ndarray): A list or array containing the losses over time.
        time_unit (str): The unit of time to be used as the x-axis of the scatter chart.

    Returns:
        Chart
    """
    # Create a range of time units from 1 to the length of losses
    tu = range(1, len(losses) + 1)
    
    # Create a DataFrame with time_unit as the x-axis and 'Losses' as the y-axis
    data = pd.DataFrame({time_unit: tu, 'Losses': losses})
    
    # Create a scatter chart using Altair
    chart = alt.Chart(data).mark_circle().encode(
        x=alt.X(time_unit),
        y='Losses',
        tooltip=[time_unit, 'Losses']
    ).properties(
        width=800,
        height=500
    )
    
    # Display the chart using Streamlit's altair_chart function
    st.altair_chart(chart, use_container_width=True)

def find_first_positive(arr):
    """
    Find the first positive value in the given array.

    Parameters:
        arr: A list or array containing numerical values.

    Returns:
        int or float or None: The first positive value found in the array. If no positive value is found, returns None.
    """
    # Loop through each number in the array
    for num in arr:
        # Check if the number is greater than zero (positive)
        if num > 0:
            # If a positive number is found, return it
            return num
    # If no positive number is found, return None
    return None


def loss_exceedance_curve(simulated_losses):
    """
    Plot the loss exceedance curve based on the simulated losses.

    Parameters:
        simulated_losses (list or numpy.ndarray): A list or array containing the simulated losses.

    Returns:
        Chart
    """
    # Calculate the losses for different percentiles
    losses = np.array([np.percentile(list(simulated_losses), x) for x in range(1, 100, 1)])
    percentiles = np.array([float(100 - x) / 100.0 for x in range(1, 100, 1)])

    # Calculate the minimum and maximum limits of the simulated losses
    x_min = find_first_positive(losses)
    x_max = max(losses)

    # Create a DataFrame to store the loss and percentile data
    data_df = pd.DataFrame({'Losses': losses, 'Percentiles': percentiles})

    # Create the Altair chart for the loss exceedance curve
    chart = alt.Chart(data_df).transform_filter(alt.datum.Losses > 0).mark_line().encode(
        x=alt.X('Losses', scale=alt.Scale(type="log", domain=[x_min, x_max]),  axis=alt.Axis(format=',.2r')),
        y=alt.Y('Percentiles', axis=alt.Axis(format='.0%'))
    )

    # Display grid lines on both axes
    chart = chart.configure_axis(grid=True)

    # Display the chart using Streamlit's altair_chart function
    st.altair_chart(chart, use_container_width=True)

def main():  
    # Hide Streamlit's default header and footer
    hide_st_style = """
    <style>
    #MainMenu {visibility : hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    st.title('SIMULACIÓN MONTECARLO')  


    # We use 'with' to create a sidebar on the left
    with st.sidebar:
        # User inputs for simulation        
        time_unit = st.selectbox('Unidad de tiempo', ['Años', 'Meses'])
        num_time_units = st.number_input('Rango de tiempo a simular', min_value=1, value=1000)
        num_events = st.number_input('Número de eventos a simular', min_value=1, value=1)

        event_data = []
        for i in range(num_events):
            st.write(f'Evento {i + 1}')
            freq = st.number_input(f'Frecuencia Evento {i + 1}', min_value=0.01,max_value=1.0, value=0.5)
            low_loss = st.number_input(f'Pérdida mínima Evento {i + 1}', value=0)
            high_loss = st.number_input(f'Pérdida máxima Evento {i + 1}', value=0)
            event_data.append({'frecuencia': freq, 'low_loss': low_loss, 'high_loss': high_loss})

            if low_loss > high_loss:
                st.warning("La Pérdida mínima no puede ser mayor que la Pérdida máxima.")
                st.stop()  # Stop further execution of the script if there's a warning

    # Check if the 'Simular' button was clicked.
    if st.sidebar.button('Simular'):
        Cp_mean ,prioritized_losses_df, df_loss_summary , losses = simulate_losses(event_data, num_time_units)
        # First plot
        st.subheader(f'Ciberpérdida Media Agregada - {time_unit}')        
        # Center the numeric value of Cp_mean
        st.markdown(f'<div style="text-align:center;font-size:2em;background-color:#b4b4b4;padding:5px;">{Cp_mean:,.1f}</div>', unsafe_allow_html=True)
        # Second plot
        st.subheader(f'Pérdidas Simuladas - {time_unit}')
        plot_scatter(losses, time_unit)
        # Third plot
        st.subheader(f'Curva de Excedencia de Pérdidas - {time_unit}')
        loss_exceedance_curve(losses)
        # Fourth plot
        st.subheader(f'Descriptivo Pérdidas - {time_unit}')
        st.table(df_loss_summary)
        # Fifth plot
        st.subheader(f'Pérdidas Priorizadas - {time_unit}')
        st.table(prioritized_losses_df[['Nombre', 'Ciberperdida Media']])

if __name__ == '__main__':
    st.set_option('deprecation.showPyplotGlobalUse', False)  # Deshabilitar la advertencia
    main()