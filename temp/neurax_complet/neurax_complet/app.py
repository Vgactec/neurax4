import streamlit as st
import numpy as np
import logging
from datetime import datetime
import json
import io
import csv
import os
from quantum_gravity_sim import QuantumGravitySimulator
from visualization import QuantumGravityVisualizer
from utils import save_simulation_data
from database import DatabaseManager
from export_manager import ExportManager
import time
import queue
import threading
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(
    page_title="Quantum Gravity Simulator",
    page_icon="🌌",
    layout="wide"
)

# Initialize session state
if 'simulator' not in st.session_state:
    st.session_state.simulator = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = QuantumGravityVisualizer()
if 'db' not in st.session_state:
    st.session_state.db = DatabaseManager()
if 'export_manager' not in st.session_state:
    st.session_state.export_manager = ExportManager()

# Title and description
st.title("🌌 Quantum Gravity Simulator")
st.markdown("""
This application simulates and visualizes quantum gravity effects in a 3D space-time grid.
Adjust the parameters below to explore different simulation conditions.
""")

# Sidebar controls
st.sidebar.header("Simulation Parameters")

# Adjusted slider parameters for better consistency
grid_size = st.sidebar.slider(
    "Grid Size",
    min_value=20,
    max_value=100,
    value=50,
    step=5,
    help="Size of the 3D space-time grid"
)

intensity = st.sidebar.slider(
    "Quantum Fluctuation Intensity",
    min_value=0.00001,
    max_value=0.001,
    value=0.0005,
    step=0.00001,
    format="%.5f",
    help="Intensity of quantum fluctuations"
)

iterations = st.sidebar.slider(
    "Simulation Steps",
    min_value=10,
    max_value=500,
    value=200,
    step=10,
    help="Number of simulation steps to run"
)

# Initialize or reset simulation
if st.sidebar.button("Initialize Simulation") or st.session_state.simulator is None:
    st.session_state.simulator = QuantumGravitySimulator(size=grid_size)
    st.success("Simulation initialized!")

# View previous simulations
st.sidebar.header("Previous Simulations")
recent_sims = st.session_state.db.get_recent_simulations()
if recent_sims:
    selected_sim = st.sidebar.selectbox(
        "View Previous Results",
        options=[(sim[0], f"Sim #{sim[0]} - {sim[1].strftime('%Y-%m-%d %H:%M')}")
                 for sim in recent_sims],
        format_func=lambda x: x[1]
    )
    if selected_sim:
        sim_data = st.session_state.db.get_simulation_by_id(selected_sim[0])
        if st.sidebar.button("Load Selected Simulation"):
            st.session_state.simulator = QuantumGravitySimulator(size=sim_data[2])
            st.session_state.simulator.space_time = np.array(sim_data[5])
            st.success(f"Loaded simulation #{sim_data[0]}")

# Main simulation controls
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Visualization")

    # Add real-time control buttons
    real_time_cols = st.columns([1, 1, 1])
    with real_time_cols[0]:
        start_sim = st.button("Start Simulation")
    with real_time_cols[1]:
        pause_sim = st.button("Pause")
    with real_time_cols[2]:
        reset_sim = st.button("Reset")

    # Add placeholder for real-time visualization
    fig_placeholder_3d = st.empty()
    fig_placeholder_2d = st.empty()

    # Initialize simulation state
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'simulation_paused' not in st.session_state:
        st.session_state.simulation_paused = False

    if start_sim:
        st.session_state.simulation_running = True
        st.session_state.simulation_paused = False

        progress_bar = st.progress(0)
        status_text = st.empty()

        while st.session_state.simulation_running and not st.session_state.simulation_paused:
            for step in range(iterations):
                if st.session_state.simulation_paused:
                    break

                # Update simulation
                st.session_state.simulator.simulate_step(intensity)

                # Update visualizations
                if step % 5 == 0:  # Update every 5 steps for performance
                    with fig_placeholder_3d:
                        fig_3d = st.session_state.visualizer.create_3d_plot(
                            st.session_state.simulator.space_time
                        )
                        st.pyplot(fig_3d)
                        plt.close(fig_3d)

                    with fig_placeholder_2d:
                        fig_2d = st.session_state.visualizer.create_slice_plot(
                            st.session_state.simulator.space_time
                        )
                        st.pyplot(fig_2d)
                        plt.close(fig_2d)

                # Update progress
                progress = (step + 1) / iterations
                progress_bar.progress(progress)
                status_text.text(f"Step {step + 1}/{iterations}")

                # Add small delay for visualization
                time.sleep(0.1)

            # Save final state to database
            metrics = st.session_state.simulator.get_metrics()
            sim_id = st.session_state.db.save_simulation(
                grid_size,
                iterations,
                intensity,
                st.session_state.simulator.space_time,
                metrics
            )
            st.success(f"Simulation #{sim_id} completed and saved to database!")
            st.session_state.simulation_running = False

    if pause_sim:
        st.session_state.simulation_paused = not st.session_state.simulation_paused
        if st.session_state.simulation_paused:
            st.info("Simulation paused")
        else:
            st.info("Simulation resumed")

    if reset_sim:
        st.session_state.simulation_running = False
        st.session_state.simulation_paused = False
        st.session_state.simulator = QuantumGravitySimulator(size=grid_size)
        st.success("Simulation reset!")


with col2:
    st.subheader("Export & Download")
    if st.session_state.simulator is not None:
        metrics = st.session_state.simulator.get_metrics()

        # Prepare simulation parameters
        parameters = {
            'grid_size': grid_size,
            'intensity': intensity,
            'iterations': iterations,
            'timestamp': datetime.now().isoformat(),
            'planck_length': st.session_state.simulator.PLANCK_LENGTH
        }

        # Export options
        st.write("📥 Choose Export Format:")

        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            if st.button("Export to Excel (.xlsx)", help="Export all data in Excel format with multiple sheets"):
                with st.spinner("Generating Excel export..."):
                    excel_file = st.session_state.export_manager.export_to_excel(
                        st.session_state.simulator.space_time,
                        metrics,
                        parameters
                    )
                    info = st.session_state.export_manager.get_export_info(excel_file)
                    with open(excel_file, 'rb') as f:
                        st.download_button(
                            label=f"📊 Download Excel ({info['size_mb']})",
                            data=f.read(),
                            file_name=os.path.basename(excel_file),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

            if st.button("Export to HDF5 (.h5)", help="Export data in HDF5 format for scientific analysis"):
                with st.spinner("Generating HDF5 export..."):
                    hdf5_file = st.session_state.export_manager.export_to_hdf5(
                        st.session_state.simulator.space_time,
                        metrics,
                        parameters
                    )
                    info = st.session_state.export_manager.get_export_info(hdf5_file)
                    with open(hdf5_file, 'rb') as f:
                        st.download_button(
                            label=f"🔬 Download HDF5 ({info['size_mb']})",
                            data=f.read(),
                            file_name=os.path.basename(hdf5_file),
                            mime="application/x-hdf5"
                        )

        with col_exp2:
            if st.button("Export Detailed CSVs", help="Export detailed data in multiple CSV files"):
                with st.spinner("Generating detailed CSV exports..."):
                    params_file, metrics_file, data_file = st.session_state.export_manager.export_to_detailed_csv(
                        st.session_state.simulator.space_time,
                        metrics,
                        parameters
                    )

                    # Create a zip file containing all CSVs
                    import zipfile
                    zip_filename = f"quantum_gravity_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                    with zipfile.ZipFile(zip_filename, 'w') as zipf:
                        zipf.write(params_file)
                        zipf.write(metrics_file)
                        zipf.write(data_file)

                    info = st.session_state.export_manager.get_export_info(zip_filename)
                    with open(zip_filename, 'rb') as f:
                        st.download_button(
                            label=f"📄 Download CSVs ({info['size_mb']})",
                            data=f.read(),
                            file_name=zip_filename,
                            mime="application/zip"
                        )

        # Display current metrics
        st.subheader("Current Metrics")
        for metric, value in metrics.items():
            st.metric(
                label=metric.replace('_', ' ').title(),
                value=f"{value:.2e}"
            )

# Footer
st.markdown("---")
st.markdown("""
Made with ❤️ using Streamlit | 
Source: [GitHub](https://github.com)
""")