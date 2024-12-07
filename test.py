from pyvis.network import Network

net = Network(notebook=True)
for n, d in H.nodes(data=True):
    net.add_node(n, label=str(n), color='blue' if d['side']=='BUY' else 'red')
for u,v in H.edges():
    net.add_edge(u,v)

net.show("graph.html")

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from ipywidgets import interact, Dropdown
from IPython.display import display

# Assume we have a list of orders after simulation:
# orders: a Python list of Order objects as defined previously (or a NumPy structured array).
# Convert them to a pandas DataFrame for inspection.

def orders_to_dataframe(orders):
    data = []
    for o in orders:
        side_str = "BUY" if o.Side == 0 else "SELL"
        state_map = {
            0: "NEW", 1: "ACK", 2: "REJECTED", 3: "PENDING_CANCEL", 
            4: "CANCELED", 5: "FILLED", 6: "PART_FILLED", 7: "PENDING_REPLACE"
        }
        data.append({
            'OrderId': o.OrderId,
            'ParentId': o.ParentId,
            'RootId': o.RootId,
            'State': state_map.get(o.State, f"Unknown({o.State})"),
            'OriginalQuantity': o.OriginalQuantity,
            'FilledQuantity': o.FilledQuantity,
            'RemainingQuantity': o.RemainingQuantity,
            'IsPing': o.IsPing,
            'TIF': o.TIF,
            'OrderType': o.OrderType,
            'Side': side_str
        })
    df = pd.DataFrame(data)
    return df

# Example: Convert orders to a DataFrame
df_orders = orders_to_dataframe(orders)  # 'orders' is your final orders list from the simulation
display(df_orders.head())  # Show the top rows

# Interactively filter by RootId using ipywidgets:
root_ids = df_orders['RootId'].unique()
@interact(root_id=Dropdown(options=sorted(root_ids), description='RootId:', continuous_update=False))
def filter_by_root(root_id):
    display(df_orders[df_orders['RootId'] == root_id])

# Similarly, we could filter by State or Side:
states = df_orders['State'].unique()
@interact(state=Dropdown(options=sorted(states), description='State:', continuous_update=False))
def filter_by_state(state):
    display(df_orders[df_orders['State'] == state])

# Visualizing Hierarchy with networkx:
# Build a directed graph where edges are ParentId -> OrderId
G = nx.DiGraph()
for _, row in df_orders.iterrows():
    oid = row['OrderId']
    pid = row['ParentId']
    G.add_node(oid, state=row['State'], side=row['Side'], ping=row['IsPing'])
    if pid != -1:
        G.add_edge(pid, oid)

# Draw the graph for a selected RootId
@interact(root_id=Dropdown(options=sorted(root_ids), description='RootId for Graph:', continuous_update=False))
def draw_graph(root_id):
    # Subgraph of orders belonging to this root
    subnodes = [n for n, d in G.nodes(data=True) if df_orders.loc[df_orders['OrderId'] == n, 'RootId'].values[0] == root_id]
    H = G.subgraph(subnodes)
    
    # Positioning nodes
    pos = nx.spring_layout(H, seed=42)  # or another layout

    # Color nodes by Side: BUY=blue, SELL=red
    node_color = []
    for n in H.nodes:
        side = df_orders.loc[df_orders['OrderId'] == n, 'Side'].values[0]
        if side == "BUY":
            node_color.append('blue')
        else:
            node_color.append('red')
    
    plt.figure(figsize=(12,8))
    nx.draw(H, pos, with_labels=True, node_color=node_color, arrows=True)
    nx.draw_networkx_labels(H, pos, labels={n: n for n in H.nodes})
    plt.title(f"Order Hierarchy for RootId={root_id}")
    plt.show()



import numpy as np

event_dtype = np.dtype([
    ('EventType', np.int8),
    ('OrderId', np.int64),
    ('ParentOrderId', np.int64),
    ('Quantity', np.int64),
    ('Timestamp', np.int64),
    ('StockSymbol', np.int64),
    ('Date', np.int64),
    ('IsPing', np.int8),
    ('Side', np.int8),
    ('TIF', np.int8),
    ('OrderType', np.int8)
])


SIDE_BUY = 0
SIDE_SELL = 1
TIF_DAY = 1
ORDER_TYPE_LIMIT = 0

events_data = np.array([
    # BUY Hierarchy (Side=0, TIF=1, OrderType=0)
    (0, 1, -1, 10000, 0,   100, 20210101, 0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O1 NEW
    (1, 1, -1, 0,     1,   100, 20210101, 0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O1 ACK
    (9, 1, -1, 2000,  2,   100, 20210101, 0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O1 FILL 2000

    (0, 2, 1, 5000,   3,   100, 20210101, 0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O2 NEW
    (1, 2, 1, 0,      4,   100, 20210101, 0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O2 ACK
    (9, 2, 1, 1000,   5,   100, 20210101, 0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O2 FILL 1000

    (0, 3, 2, 5000,   6,   100, 20210101, 0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O3 NEW
    (1, 3, 2, 0,      7,   100, 20210101, 0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O3 ACK
    (9, 3, 2, 500,    8,   100, 20210101, 0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O3 FILL 500

    (0, 4, 3, 2000,   9,   100, 20210101, 0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O4 NEW
    (1, 4, 3, 0,      10,  100, 20210101, 0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O4 ACK
    (9, 4, 3, 500,    11,  100, 20210101, 0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O4 FILL 500

    (0, 5, 3, 2000,   12,  100, 20210101, 0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O5 NEW
    (1, 5, 3, 0,      13,  100, 20210101, 0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O5 ACK
    (3, 5, 3, 0,      14,  100, 20210101, 0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O5 CANCEL_NEW
    (4, 5, 3, 0,      15,  100, 20210101, 0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O5 CANCEL_ACK

    (0, 8, 4, 1000,   16,  100, 20210101, 1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O8 NEW ping
    (1, 8, 4, 0,      17,  100, 20210101, 1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O8 ACK
    (9, 8, 4, 1000,   18,  100, 20210101, 1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O8 FULL FILL

    (0, 9, 4, 1000,   19,  100, 20210101, 1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O9 NEW ping
    (1, 9, 4, 0,      20,  100, 20210101, 1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O9 ACK
    (9, 9, 4, 200,    21,  100, 20210101, 1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT), # O9 PARTIAL FILL 200

    (0, 10,5, 1000,   22,  100, 20210101,1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # O10 NEW ping
    (1, 10,5, 0,      23,  100, 20210101,1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # O10 ACK
    (9, 10,5,1000,    24,  100, 20210101,1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # O10 FULL FILL

    (0, 11,5,1000,    25,  100, 20210101,1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # O11 NEW ping
    (1, 11,5,0,       26,  100, 20210101,1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # O11 ACK
    (3, 11,5,0,       27,  100, 20210101,1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # O11 CANCEL_NEW
    (4, 11,5,0,       28,  100, 20210101,1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # O11 CANCEL_ACK

    (0, 6, 3, 2000,   29,  100, 20210101,0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # O6 NEW
    (1, 6, 3, 0,      30,  100, 20210101,0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # O6 ACK
    (9, 6, 3, 400,    31,  100, 20210101,0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # O6 PART FILL 400

    (0, 7, 3, 2000,   32,  100, 20210101,0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # O7 NEW
    (1, 7, 3, 0,      33,  100, 20210101,0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # O7 ACK
    (6, 7, 3, 2500,   34,  100, 20210101,0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # O7 REPLACE_NEW(2500)
    (7, 7, 3, 0,      35,  100, 20210101,0, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # O7 REPLACE_ACK

    (0, 12,6,1000,    36,  100, 20210101,1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # O12 ping
    (1, 12,6,0,       37,  100, 20210101,1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # O12 ACK
    (9, 12,6,300,     38,  100, 20210101,1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # O12 PART FILL 300

    (0, 13,6,1000,    39,  100, 20210101,1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),
    (1, 13,6,0,       40,  100, 20210101,1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),
    (9, 13,6,1000,    41,  100, 20210101,1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # O13 FULL FILL

    (0, 14,7,1000,    42,  100, 20210101,1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),
    (1, 14,7,0,       43,  100, 20210101,1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),
    (3, 14,7,0,       44,  100, 20210101,1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # CANCEL_NEW
    (4, 14,7,0,       45,  100, 20210101,1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # CANCEL_ACK

    (0, 15,7,1000,    46,  100, 20210101,1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),
    (1, 15,7,0,       47,  100, 20210101,1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),
    (9, 15,7,500,     48,  100, 20210101,1, SIDE_BUY, TIF_DAY, ORDER_TYPE_LIMIT),  # PART FILL 500

    # SELL Hierarchy (Side=1)
    (0,20,-1,8000,    49, 100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),   
    (1,20,-1,0,        50,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (9,20,-1,1000,     51,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),   # PART FILL 1000

    (0,21,20,4000,     52,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (1,21,20,0,        53,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (9,21,20,800,      54,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),   # PART FILL 800

    (0,22,20,4000,     55,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (1,22,20,0,        56,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (9,22,20,600,      57,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),   # PART FILL 600

    (0,23,22,2000,     58,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (1,23,22,0,        59,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (9,23,22,300,      60,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),   # PART FILL 300

    (0,24,22,2000,     61,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (1,24,22,0,        62,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (3,24,22,0,        63,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),   # CANCEL_NEW
    (4,24,22,0,        64,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),   # CANCEL_ACK

    (0,25,23,1000,     65,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (1,25,23,0,        66,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (9,25,23,1000,     67,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),   # FULL FILL

    (0,26,23,1000,     68,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (1,26,23,0,        69,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (9,26,23,100,      70,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),   # PART FILL 100

    (0,27,24,1000,     71,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (1,27,24,0,        72,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (9,27,24,1000,     73,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),   # FULL FILL

    (0,28,24,1000,     74,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (1,28,24,0,        75,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (9,28,24,200,      76,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),   # PART FILL 200

    (0,29,22,2000,     77,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (1,29,22,0,        78,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (6,29,22,1800,     79,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),   # REPLACE_NEW(1800)
    (7,29,22,0,        80,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),   # REPLACE_ACK

    (0,30,22,2000,     81,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (1,30,22,0,        82,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (9,30,22,400,      83,100,20210101,0, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),   # PART FILL 400

    (0,31,29,1000,     84,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (1,31,29,0,        85,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (3,31,29,0,        86,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),   # CANCEL_NEW
    (4,31,29,0,        87,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),   # CANCEL_ACK

    (0,32,29,1000,     88,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (1,32,29,0,        89,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (9,32,29,1000,     90,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),   # FULL FILL

    (0,33,30,1000,     91,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (1,33,30,0,        92,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (9,33,30,300,      93,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),   # PART FILL 300

    (0,34,30,1000,     94,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (1,34,30,0,        95,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT),
    (9,34,30,1000,     96,100,20210101,1, SIDE_SELL, TIF_DAY, ORDER_TYPE_LIMIT)    # FULL FILL
], dtype=event_dtype)

