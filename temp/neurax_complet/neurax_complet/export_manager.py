import pandas as pd
import h5py
import numpy as np
import json
import os
from datetime import datetime
import logging

class ExportManager:
    def __init__(self):
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        
    def export_to_excel(self, space_time, metrics, parameters):
        """Export simulation data to Excel format with multiple sheets"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quantum_gravity_export_{timestamp}.xlsx"
        
        with pd.ExcelWriter(filename) as writer:
            # Parameters sheet
            pd.DataFrame([parameters]).to_excel(writer, sheet_name='Parameters', index=False)
            
            # Metrics sheet
            pd.DataFrame([metrics]).to_excel(writer, sheet_name='Metrics', index=False)
            
            # Space-time data sheet (flattened with coordinates)
            z, y, x = np.where(space_time != 0)  # Only non-zero values
            values = space_time[z, y, x]
            space_time_df = pd.DataFrame({
                'x': x, 'y': y, 'z': z,
                'value': values
            })
            space_time_df.to_excel(writer, sheet_name='Space-Time Data', index=False)
        
        logging.info(f"Exported data to Excel: {filename}")
        return filename

    def export_to_hdf5(self, space_time, metrics, parameters):
        """Export simulation data to HDF5 format for scientific analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quantum_gravity_data_{timestamp}.h5"
        
        with h5py.File(filename, 'w') as f:
            # Create groups
            sim_group = f.create_group('simulation')
            param_group = f.create_group('parameters')
            metrics_group = f.create_group('metrics')
            
            # Store space-time data
            sim_group.create_dataset('space_time', data=space_time)
            
            # Store parameters
            for key, value in parameters.items():
                param_group.attrs[key] = value
            
            # Store metrics
            for key, value in metrics.items():
                metrics_group.attrs[key] = value
                
        logging.info(f"Exported data to HDF5: {filename}")
        return filename

    def export_to_detailed_csv(self, space_time, metrics, parameters):
        """Export detailed simulation data to CSV format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"quantum_gravity_detailed_{timestamp}"
        
        # Export parameters
        params_file = f"{base_filename}_parameters.csv"
        pd.DataFrame([parameters]).to_csv(params_file, index=False)
        
        # Export metrics
        metrics_file = f"{base_filename}_metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
        
        # Export space-time data
        data_file = f"{base_filename}_data.csv"
        z, y, x = np.where(space_time != 0)  # Only non-zero values
        values = space_time[z, y, x]
        pd.DataFrame({
            'x': x, 'y': y, 'z': z,
            'value': values,
            'timestamp': datetime.now().isoformat()
        }).to_csv(data_file, index=False)
        
        logging.info(f"Exported detailed CSV files with base name: {base_filename}")
        return params_file, metrics_file, data_file

    def get_export_info(self, filename):
        """Get file size and format information"""
        size_bytes = os.path.getsize(filename)
        size_mb = size_bytes / (1024 * 1024)
        return {
            'filename': filename,
            'size_mb': f"{size_mb:.2f} MB",
            'format': filename.split('.')[-1].upper()
        }
