import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
from ipytree import Tree, Node
import networkx as nx
import matplotlib.pyplot as plt
from ipydatagrid import DataGrid

# Global simulation state
orders = []  # Will hold Order objects
order_id_to_index = {}
parent_to_children = {}

current_event_index = 0  # which event we are processed up to
max_event_index = 0

# Visualization widgets
grid_display = None
tree_display = widgets.Output()
graph_display = widgets.Output()

# Assume we have a function orders_to_dataframe(orders) to convert orders to a DataFrame
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
    return pd.DataFrame(data)

# Step through events one at a time
def step_event(events, index):
    """
    Process the event at `events[index]` on the global orders state.
    This function is similar to a single iteration of the process_events loop.
    """
    if index < 0 or index >= events.shape[0]:
        return False
    etype = events[index]['EventType']
    oid = events[index]['OrderId']
    poid = events[index]['ParentOrderId']
    qty = events[index]['Quantity']
    stock_symbol = events[index]['StockSymbol']
    date = events[index]['Date']
    is_ping = (events[index]['IsPing'] != 0)
    side = events[index]['Side']
    tif = events[index]['TIF']
    order_type = events[index]['OrderType']

    # Logic for handling events
    # Similar to what you have; not reproduced here for brevity.
    # Implement event handling as per your full process_events logic.
    pass

# Reset simulation state
def reset_simulation(events):
    global orders, order_id_to_index, parent_to_children, current_event_index, max_event_index
    orders = []
    order_id_to_index = {}
    parent_to_children = {}
    current_event_index = 0
    max_event_index = events.shape[0]

# Run events up to a given index
def run_to_event(events, target_index):
    global current_event_index
    if target_index < current_event_index:
        reset_simulation(events)
    while current_event_index <= target_index and current_event_index < max_event_index:
        step_event(events, current_event_index)
        current_event_index += 1

# Update visuals after processing an event
def update_visuals():
    global grid_display
    df = orders_to_dataframe(orders)

    # Update the DataGrid
    grid_display.dataframe = df

    # Update the tree display
    with tree_display:
        tree_display.clear_output()
        if len(df) > 0:
            root_ids = df['RootId'].unique()
            if len(root_ids) > 0:
                root_id = root_ids[0]
                df_sub = df[df['RootId'] == root_id]
                tree = Tree()
                node = build_tree_from_df(df_sub, root_id)
                tree.add_node(node)
                display(tree)

    # Update the graph display
    with graph_display:
        graph_display.clear_output()
        G = build_graph_from_df(df)
        fig = plot_graph(G, df)
        display(fig)

# Build a tree from a DataFrame
def build_tree_from_df(df, root_id):
    child_map = {}
    for _, row in df.iterrows():
        pid = row['ParentId']
        if pid not in child_map:
            child_map[pid] = []
        child_map[pid].append(row)

    def create_node(order):
        label = f"OrderId={order['OrderId']} State={order['State']}"
        node = Node(label)
        oid = order['OrderId']
        if oid in child_map:
            for child_order in child_map[oid]:
                node.add_node(create_node(child_order))
        return node

    root_orders = df[df['ParentId'] == -1]
    root_order = root_orders[root_orders['RootId'] == root_id].iloc[0]
    return create_node(root_order)

# Build a graph from a DataFrame
def build_graph_from_df(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        oid = row['OrderId']
        pid = row['ParentId']
        G.add_node(oid)
        if pid != -1:
            G.add_edge(pid, oid)
    return G

# Plot the graph
def plot_graph(G, df):
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True, ax=ax)
    return fig

# Initialize the DataGrid
def initialize_grid():
    df = orders_to_dataframe(orders)
    grid = DataGrid(df, selection_mode="row", layout={"height": "300px", "width": "100%"})
    return grid

# Slider to step through events
event_slider = widgets.IntSlider(value=0, min=0, max=0, step=1, description='Event Index', continuous_update=False)

def on_slider_change(change):
    if change['name'] == 'value':
        target_index = change['new']
        run_to_event(events_data, target_index)
        update_visuals()

event_slider.observe(on_slider_change, 'value')

# Initialize everything
events_data = ... # Your events_data
reset_simulation(events_data)
event_slider.max = len(events_data) - 1

grid_display = initialize_grid()

# Display the dashboard
dashboard = widgets.VBox([
    widgets.HTML("<h3>Step Through Events</h3>"),
    event_slider,
    grid_display,
    tree_display,
    graph_display
])
display(dashboard)

# Initial update
update_visuals()
