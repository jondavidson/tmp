import pykx as kx
from queue import Queue

class KdbConnectionPool:
    def __init__(self, host, port, size=5):
        self.pool = Queue(maxsize=size)
        for _ in range(size):
            self.pool.put(kx.SyncQConnection(host=host, port=port))

    def get_connection(self):
        return self.pool.get()

    def release_connection(self, conn):
        self.pool.put(conn)

    def close_all(self):
        while not self.pool.empty():
            conn = self.pool.get()
            conn.close()

# Initialize the pool
pool = KdbConnectionPool(host='localhost', port=5000, size=5)

try:
    # Acquire a connection
    conn = pool.get_connection()
    result = conn('{x+y}', 2, 3)
    print(result)
    # Release the connection back to the pool
    pool.release_connection(conn)
finally:
    # Close all connections in the pool
    pool.close_all()
