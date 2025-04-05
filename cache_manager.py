import sqlite3
import hashlib
import time
import os
import json
import logging
import numpy as np # Added for embedding serialization/deserialization
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)

# Helper function to create a stable hash from potentially complex config dicts
def hash_config(config: Dict[str, Any]) -> str:
    """Creates a stable SHA256 hash from a dictionary."""
    # Sort the dictionary by key to ensure consistent order
    # Filter out None values to avoid inconsistencies if a key is sometimes absent
    filtered_config = {k: v for k, v in config.items() if v is not None}
    sorted_config_str = json.dumps(filtered_config, sort_keys=True)
    return hashlib.sha256(sorted_config_str.encode('utf-8')).hexdigest()

def hash_text(text: str) -> str:
    """Creates a SHA256 hash from a string."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

class CacheManager:
    def __init__(self, db_path: str = "cache/nanosage_cache.db"):
        self.db_path = db_path
        self._ensure_dir_exists()
        self.conn = None
        try:
            # Using check_same_thread=False for potential use in threaded GUI environments
            # Consider implications if using heavy multiprocessing later
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._create_tables()
            logger.info(f"Cache initialized at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to or initializing cache database: {e}")
            self.conn = None # Ensure connection is None if setup fails

    def _ensure_dir_exists(self):
        """Ensures the directory for the database file exists."""
        dir_name = os.path.dirname(self.db_path)
        if dir_name and not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
                logger.info(f"Created cache directory: {dir_name}")
            except OSError as e:
                logger.error(f"Error creating cache directory {dir_name}: {e}")

    def _create_tables(self):
        """Creates necessary tables if they don't exist."""
        if not self.conn: return
        try:
            cursor = self.conn.cursor()
            # Web Cache
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS web_cache (
                    url_hash TEXT PRIMARY KEY,
                    url TEXT,
                    content TEXT,
                    timestamp INTEGER
                )
            ''')
            # Embedding Cache
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    text_hash TEXT,
                    model_name TEXT,
                    embedding BLOB,
                    timestamp INTEGER,
                    PRIMARY KEY (text_hash, model_name)
                )
            ''')
            # Summary Cache
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS summary_cache (
                    text_hash TEXT,
                    model_config_hash TEXT,
                    summary TEXT,
                    timestamp INTEGER,
                    PRIMARY KEY (text_hash, model_config_hash)
                )
            ''')
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error creating cache tables: {e}")

    def clear_all_cache(self):
        """Deletes and recreates the cache database file."""
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
                if os.path.exists(self.db_path):
                    os.remove(self.db_path)
                    logger.info(f"Cache database deleted: {self.db_path}")
                # Re-initialize
                self.__init__(self.db_path) # Reconnect and create tables
            except OSError as e:
                logger.error(f"Error deleting cache file {self.db_path}: {e}")
            except sqlite3.Error as e:
                 logger.error(f"Error during cache clearing: {e}")


    # --- Web Cache Methods ---
    def get_web_content(self, url: str) -> Optional[str]:
        if not self.conn: return None
        url_hash = hash_text(url)
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT content FROM web_cache WHERE url_hash = ?", (url_hash,))
            result = cursor.fetchone()
            if result:
                logger.debug(f"Cache hit for web content: {url}")
                return result[0]
            logger.debug(f"Cache miss for web content: {url}")
            return None
        except sqlite3.Error as e:
            logger.warning(f"Error reading web cache for {url}: {e}")
            return None

    def store_web_content(self, url: str, content: str):
        if not self.conn: return
        url_hash = hash_text(url)
        timestamp = int(time.time())
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO web_cache (url_hash, url, content, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (url_hash, url, content, timestamp))
            self.conn.commit()
            logger.debug(f"Stored web content in cache: {url}")
        except sqlite3.Error as e:
            logger.warning(f"Error writing web cache for {url}: {e}")

    # --- Embedding Cache Methods ---
    def get_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        if not self.conn: return None
        text_hash = hash_text(text)
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT embedding FROM embedding_cache
                WHERE text_hash = ? AND model_name = ?
            ''', (text_hash, model_name))
            result = cursor.fetchone()
            if result and result[0]:
                logger.debug(f"Cache hit for embedding: model={model_name}, text_hash={text_hash[:8]}...")
                # Deserialize bytes back to numpy array
                embedding = np.frombuffer(result[0], dtype=np.float32) # Assuming float32, adjust if needed
                return embedding
            logger.debug(f"Cache miss for embedding: model={model_name}, text_hash={text_hash[:8]}...")
            return None
        except sqlite3.Error as e:
            logger.warning(f"Error reading embedding cache: {e}")
            return None
        except Exception as e: # Catch potential deserialization errors
             logger.warning(f"Error deserializing embedding from cache: {e}")
             return None

    def store_embedding(self, text: str, model_name: str, embedding: np.ndarray):
        if not self.conn: return
        text_hash = hash_text(text)
        timestamp = int(time.time())
        try:
            # Serialize embedding to bytes
            embedding_bytes = embedding.astype(np.float32).tobytes() # Assuming float32
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO embedding_cache (text_hash, model_name, embedding, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (text_hash, model_name, embedding_bytes, timestamp))
            self.conn.commit()
            logger.debug(f"Stored embedding in cache: model={model_name}, text_hash={text_hash[:8]}...")
        except sqlite3.Error as e:
            logger.warning(f"Error writing embedding cache: {e}")
        except Exception as e: # Catch potential serialization errors
            logger.warning(f"Error serializing embedding for cache: {e}")


    # --- Summary Cache Methods ---
    def get_summary(self, text: str, model_config: Dict[str, Any]) -> Optional[str]:
        if not self.conn: return None
        text_hash = hash_text(text)
        config_hash = hash_config(model_config)
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT summary FROM summary_cache
                WHERE text_hash = ? AND model_config_hash = ?
            ''', (text_hash, config_hash))
            result = cursor.fetchone()
            if result:
                logger.debug(f"Cache hit for summary: config_hash={config_hash[:8]}, text_hash={text_hash[:8]}...")
                return result[0]
            logger.debug(f"Cache miss for summary: config_hash={config_hash[:8]}, text_hash={text_hash[:8]}...")
            return None
        except sqlite3.Error as e:
            logger.warning(f"Error reading summary cache: {e}")
            return None

    def store_summary(self, text: str, model_config: Dict[str, Any], summary: str):
        if not self.conn: return
        text_hash = hash_text(text)
        config_hash = hash_config(model_config)
        timestamp = int(time.time())
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO summary_cache (text_hash, model_config_hash, summary, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (text_hash, config_hash, summary, timestamp))
            self.conn.commit()
            logger.debug(f"Stored summary in cache: config_hash={config_hash[:8]}, text_hash={text_hash[:8]}...")
        except sqlite3.Error as e:
            logger.warning(f"Error writing summary cache: {e}")

    def close(self):
        """Closes the database connection."""
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
                logger.info("Cache connection closed.")
            except sqlite3.Error as e:
                logger.error(f"Error closing cache connection: {e}")
