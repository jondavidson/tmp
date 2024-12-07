import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import qgrid
from ipytree import Tree, Node
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Assume we already have:
# - event_dtype and events_data as defined previously
# - The Order class and helper functions like propagate_up_down from before
# - A function orders_to_dataframe(orders) from previous examples

# Global simulation state
orders = []  # Will hold Order objects
order_id_to_index = {}
parent_to_children = {}

current_event_index = 0  # which event we are processed up to
max_event_index = 0

# For visualization widgets
df_display = widgets.Output()
graph_display = widgets.Output()
tree_display = widgets.Output()

# We assume we have a function `step_event` that processes one event at a time
def step_event(events, index):
    """
    Process the event at `events[index]` on the global orders state.
    This function is similar to a single iteration of the process_events loop.
    Returns True if an event was processed, False if index out of range.
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

    # Similar logic as in the full process_events, but just for one event:
    event_affects_idx = -1

    if etype == 0: # EVENT_ORDER_NEW
        if poid in order_id_to_index:
            p_idx = order_id_to_index[poid]
            root_id = orders[p_idx].RootId
        else:
            p_idx = -1
            root_id = len(orders)
        from numba.typed import List
        # Create new order
        o = Order(oid, p_idx, root_id, 0, qty, 0, qty, stock_symbol, date, is_ping, tif, order_type, side)
        orders.append(o)
        order_id_to_index[oid] = len(orders)-1
        event_affects_idx = len(orders)-1
        if poid != -1:
            if poid not in parent_to_children:
                parent_to_children[poid] = []
            parent_to_children[poid].append(oid)

    else:
        # Existing order required
        if oid not in order_id_to_index:
            # invalid event
            return True
        idx = order_id_to_index[oid]
        o = orders[idx]

        old_state = o.State
        old_filled = o.FilledQuantity
        old_rem = o.RemainingQuantity
        old_orig = o.OriginalQuantity

        # The logic here should mirror the previously shown event handling
        # For brevity, show a couple of cases:
        if etype == 1: # EVENT_ORDER_ACK
            if o.State == 0: # NEW
                o.State = 1 # ACK
                event_affects_idx = idx
        elif etype == 9: # EVENT_FILL
            if o.State in (1,6): # ACK or PART_FILLED
                fill_qty = qty
                if fill_qty > o.RemainingQuantity:
                    fill_qty = o.RemainingQuantity
                o.FilledQuantity += fill_qty
                o.RemainingQuantity -= fill_qty
                if o.RemainingQuantity == 0:
                    o.State = 5 # FILLED
                else:
                    if o.State not in (4,5,2):
                        o.State = 6 # PART_FILLED
                event_affects_idx = idx
        # ... Implement other event handlers similarly ...

    # If we updated something, propagate_up_down
    if event_affects_idx != -1:
        logs = propagate_up_down(orders, order_id_to_index, parent_to_children, event_affects_idx)
        # We could log or ignore these logs

    return True

def reset_simulation(events):
    global orders, order_id_to_index, parent_to_children, current_event_index, max_event_index
    orders = []
    order_id_to_index = {}
    parent_to_children = {}
    current_event_index = 0
    max_event_index = events.shape[0]

def run_to_event(events, target_index):
    # Process events up to target_index
    global current_event_index
    if target_index < current_event_index:
        # If going backwards, reset and re-run
        reset_simulation(events)
    while current_event_index <= target_index and current_event_index < max_event_index:
        step_event(events, current_event_index)
        current_event_index += 1

def update_visuals():
    # Convert orders to dataframe and update qgrid and tree/graph
    df = orders_to_dataframe(orders)
    with df_display:
        df_display.clear_output()
        display(df)

    # Update the hierarchy tree (just show one root for demo)
    with tree_display:
        tree_display.clear_output()
        if len(df)==0:
            return
        root_ids = df['RootId'].unique()
        if len(root_ids)>0:
            root_id = root_ids[0]
            df_sub = df[df['RootId']==root_id]
            tree = Tree()
            node = build_tree_from_df(df_sub, root_id)
            tree.add_node(node)
            display(tree)

    # Update a simple graph visualization:
    with graph_display:
        graph_display.clear_output()
        G = build_graph_from_df(df)
        fig = plot_graph(G, df)
        display(fig)

def build_tree_from_df(df, root_id):
    # Similar logic as before
    child_map = {}
    for _,row in df.iterrows():
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

    root_orders = df[df['ParentId']==-1]
    root_order = root_orders[root_orders['RootId']==root_id].iloc[0]
    return create_node(root_order)

def build_graph_from_df(df):
    G = nx.DiGraph()
    for _,row in df.iterrows():
        oid = row['OrderId']
        pid = row['ParentId']
        G.add_node(oid)
        if pid != -1:
            G.add_edge(pid, oid)
    return G

def plot_graph(G, df):
    # A simple matplotlib plot
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(12,8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True, ax=ax)
    return fig

# Create a slider to step through events
event_slider = widgets.IntSlider(value=0, min=0, max=0, step=1, description='Event Index', continuous_update=False)

def on_slider_change(change):
    if change['name'] == 'value':
        target_index = change['new']
        run_to_event(events_data, target_index)
        update_visuals()

event_slider.observe(on_slider_change, 'value')

# Initialize simulation
events_data = ... # Your events_data from before with the extended dtype
reset_simulation(events_data)
event_slider.max = len(events_data)-1

# Display the dashboard
dashboard = widgets.VBox([
    widgets.HTML("<h3>Step Through Events</h3>"),
    event_slider,
    widgets.HBox([df_display, tree_display]),
    graph_display
])
display(dashboard)
update_visuals()  # initial empty state
