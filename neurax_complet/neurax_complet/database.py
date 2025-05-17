import os
import psycopg2
from psycopg2.extras import Json
import numpy as np
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class DatabaseManager:
    def __init__(self, fallback_file="local_database_fallback.json"):
        self.use_fallback = False
        self.fallback_data = {"simulations": []}
        self.fallback_file = fallback_file
        
        try:
            self.conn = psycopg2.connect(os.environ.get('DATABASE_URL', ''))
            self.create_tables()
            logging.info("Database connection established")
        except Exception as e:
            logging.error(f"Database connection failed: {e}")
            self.use_fallback = True
            self._load_fallback_data()
            logging.warning("Using local JSON fallback for database operations")

    def _load_fallback_data(self):
        """Load local JSON data as fallback database"""
        try:
            if os.path.exists(self.fallback_file):
                with open(self.fallback_file, 'r') as f:
                    self.fallback_data = json.load(f)
                logging.info(f"Loaded fallback data from {self.fallback_file}")
            else:
                self.fallback_data = {"simulations": []}
                logging.info("Created new fallback database")
        except Exception as e:
            logging.error(f"Error loading fallback data: {e}")
            self.fallback_data = {"simulations": []}
    
    def _save_fallback_data(self):
        """Save local JSON data"""
        try:
            with open(self.fallback_file, 'w') as f:
                json.dump(self.fallback_data, f, indent=2)
            logging.info(f"Saved fallback data to {self.fallback_file}")
        except Exception as e:
            logging.error(f"Error saving fallback data: {e}")
            
    def create_tables(self):
        """Create database tables or prepare fallback structure"""
        if self.use_fallback:
            logging.info("Using fallback database - no tables to create")
            return
            
        with self.conn.cursor() as cur:
            # Create simulations table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS simulations (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    grid_size INTEGER,
                    iterations INTEGER,
                    intensity FLOAT,
                    space_time_data JSONB,
                    metrics JSONB
                )
            ''')
            self.conn.commit()
            logging.info("Database tables created successfully")

    def save_simulation(self, grid_size, iterations, intensity, space_time, metrics):
        """Save simulation data to database or fallback JSON"""
        if self.use_fallback:
            # Generate a timestamp
            timestamp = datetime.now().isoformat()
            
            # Generate an ID (incremental based on existing data)
            sim_id = 1
            if self.fallback_data["simulations"]:
                sim_id = max(sim["id"] for sim in self.fallback_data["simulations"]) + 1
                
            # Create simulation record
            simulation = {
                "id": sim_id,
                "timestamp": timestamp,
                "grid_size": grid_size,
                "iterations": iterations,
                "intensity": float(intensity),
                "space_time_data": space_time.tolist(),
                "metrics": metrics
            }
            
            self.fallback_data["simulations"].append(simulation)
            self._save_fallback_data()
            
            logging.info(f"Simulation data saved to fallback with ID: {sim_id}")
            return sim_id
        else:
            with self.conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO simulations 
                    (grid_size, iterations, intensity, space_time_data, metrics)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                ''', (
                    grid_size,
                    iterations,
                    intensity,
                    Json(space_time.tolist()),
                    Json(metrics)
                ))
                simulation_id = cur.fetchone()[0]
                self.conn.commit()
                logging.info(f"Simulation data saved with ID: {simulation_id}")
                return simulation_id

    def get_recent_simulations(self, limit=5):
        """Get recent simulations from database or fallback JSON"""
        if self.use_fallback:
            # Sort by timestamp (recent first) and limit
            sorted_sims = sorted(
                self.fallback_data["simulations"], 
                key=lambda x: x["timestamp"], 
                reverse=True
            )[:limit]
            
            # Convert to format similar to database results
            result = []
            for sim in sorted_sims:
                # Format as (id, timestamp, grid_size, iterations, intensity, metrics)
                result.append((
                    sim["id"],
                    datetime.fromisoformat(sim["timestamp"]),
                    sim["grid_size"],
                    sim["iterations"],
                    sim["intensity"],
                    sim["metrics"]
                ))
            
            return result
        else:
            with self.conn.cursor() as cur:
                cur.execute('''
                    SELECT id, timestamp, grid_size, iterations, intensity, metrics
                    FROM simulations
                    ORDER BY timestamp DESC
                    LIMIT %s
                ''', (limit,))
                return cur.fetchall()

    def get_simulation_by_id(self, simulation_id):
        """Get a specific simulation by ID from database or fallback JSON"""
        if self.use_fallback:
            # Find the simulation with the given ID
            for sim in self.fallback_data["simulations"]:
                if sim["id"] == simulation_id:
                    # Convert to format similar to database results
                    return (
                        sim["id"],
                        datetime.fromisoformat(sim["timestamp"]),
                        sim["grid_size"],
                        sim["iterations"],
                        sim["intensity"],
                        sim["space_time_data"],
                        sim["metrics"]
                    )
            return None
        else:
            with self.conn.cursor() as cur:
                cur.execute('''
                    SELECT * FROM simulations WHERE id = %s
                ''', (simulation_id,))
                return cur.fetchone()

    def close(self):
        """Close database connection if using real database"""
        if not self.use_fallback:
            self.conn.close()
            logging.info("Database connection closed")
        else:
            # Save any pending changes to fallback data
            self._save_fallback_data()
            logging.info("Fallback database saved")
