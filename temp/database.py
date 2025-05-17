import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    pass

class JsonStorage:
    def __init__(self):
        logger.info("Using JSON storage")

    def save(self, data):
        # Placeholder for saving to JSON file
        print("Saving data to JSON file")

    def load(self):
        # Placeholder for loading from JSON file
        print("Loading data from JSON file")
        return {}

class DatabaseStorage:
    def __init__(self):
        try:
            # Simulate a database connection
            self.connection = self.connect_to_database()
            logger.info("Connected to database")
        except DatabaseError:
            self.storage = JsonStorage() # Fallback to JSON
        else:
            self.storage = None # Database is available

    def connect_to_database(self):
        # Simulate database connection failure
        raise DatabaseError("Connection failed")
        # In real code, it would be:
        # try:
        #   return psycopg2.connect(...)
        # except Exception as e:
        #   raise DatabaseError("Connection failed") from e

    def save(self, data):
        if self.storage:
            self.storage.save(data)
        else:
            # Save to database
            print("Saving data to database")

    def load(self):
        if self.storage:
            return self.storage.load()
        else:
            # Load from database
            print("Loading data from database")
            return {}

# --- Usage Example ---
if __name__ == "__main__":
    try:
        db = DatabaseStorage()
        data = db.load()
        db.save({"key": "value"})
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        print("Finished")