import numpy as np
from numba import njit, int64, float64, int8, boolean, types
from numba.typed import List, Dict

# Define constants for order states
STATE_PENDING = 0
STATE_ACK = 1
STATE_PENDING_CANCEL = 2
STATE_CANCELED = 3
STATE_FILLED = 4

# Define constants for event types
EVENT_NEW_ORDER = 0
EVENT_FILL = 1
EVENT_CANCEL = 2
EVENT_PING_ORDER = 3

# Define order data structure
order_dtype = np.dtype([
    ('OrderId', np.int64),
    ('ParentIndex', np.int64),
    ('RootIndex', np.int64),
    ('State', np.int8),
    ('OriginalQuantity', np.float64),
    ('FilledQuantity', np.float64),
    ('RemainingQuantity', np.float64),
    ('StockSymbol', np.int64),  # Assume StockSymbol is mapped to an integer
    ('Date', np.int64),         # Assume Date is represented as an integer (e.g., YYYYMMDD)
    ('IsPing', np.bool_)
])

# Define event data structure
event_dtype = np.dtype([
    ('EventType', np.int8),
    ('OrderId', np.int64),
    ('ParentOrderId', np.int64),
    ('Quantity', np.float64),
    ('Timestamp', np.int64),
    ('StockSymbol', np.int64),
    ('Date', np.int64),
    ('IsPing', np.bool_)
])

@njit
def create_order(OrderId, ParentIndex, RootIndex, State, OriginalQuantity,
                 FilledQuantity, RemainingQuantity, StockSymbol, Date, IsPing):
    order = np.empty(1, dtype=order_dtype)[0]
    order['OrderId'] = OrderId
    order['ParentIndex'] = ParentIndex
    order['RootIndex'] = RootIndex
    order['State'] = State
    order['OriginalQuantity'] = OriginalQuantity
    order['FilledQuantity'] = FilledQuantity
    order['RemainingQuantity'] = RemainingQuantity
    order['StockSymbol'] = StockSymbol
    order['Date'] = Date
    order['IsPing'] = IsPing
    return order

@njit
def process_events(events, orders, order_id_to_index, market_impact):
    for i in range(len(events)):
        event = events[i]
        event_type = event['EventType']

        if event_type == EVENT_NEW_ORDER or event_type == EVENT_PING_ORDER:
            handle_new_order(event, orders, order_id_to_index)
        elif event_type == EVENT_FILL:
            handle_fill(event, orders, order_id_to_index, market_impact)
        elif event_type == EVENT_CANCEL:
            handle_cancel(event, orders, order_id_to_index)
        else:
            # Unknown event type
            continue

@njit
def handle_new_order(event, orders, order_id_to_index):
    order_id = event['OrderId']
    parent_order_id = event['ParentOrderId']
    quantity = event['Quantity']
    is_ping = event['IsPing']

    # Find parent index
    parent_index = -1
    if parent_order_id in order_id_to_index:
        parent_index = order_id_to_index[parent_order_id]

    # Determine root index
    if parent_index != -1:
        root_index = orders[parent_index]['RootIndex']
    else:
        root_index = len(orders)  # Self root if no parent

    # Create new order
    order = create_order(
        OrderId=order_id,
        ParentIndex=parent_index,
        RootIndex=root_index,
        State=STATE_ACK,  # Assuming orders are acknowledged upon creation
        OriginalQuantity=quantity,
        FilledQuantity=0.0,
        RemainingQuantity=quantity,
        StockSymbol=event['StockSymbol'],
        Date=event['Date'],
        IsPing=is_ping
    )

    # Add to orders list and mapping
    orders.append(order)
    order_id_to_index[order_id] = len(orders) - 1

@njit
def handle_fill(event, orders, order_id_to_index, market_impact):
    order_id = event['OrderId']
    quantity = event['Quantity']

    if order_id in order_id_to_index:
        index = order_id_to_index[order_id]
        order = orders[index]

        # Calculate fill quantity
        fill_quantity = min(order['RemainingQuantity'], quantity)
        order['FilledQuantity'] += fill_quantity
        order['RemainingQuantity'] -= fill_quantity

        # Update state if fully filled
        if order['RemainingQuantity'] == 0.0:
            order['State'] = STATE_FILLED

        # Update market impact
        key = (order['StockSymbol'], order['Date'])
        if key in market_impact:
            market_impact[key] += fill_quantity
        else:
            market_impact[key] = fill_quantity

        # Propagate fill up the hierarchy
        propagate_fill_up(orders, index, fill_quantity)

@njit
def handle_cancel(event, orders, order_id_to_index):
    order_id = event['OrderId']

    if order_id in order_id_to_index:
        index = order_id_to_index[order_id]
        order = orders[index]
        order['State'] = STATE_CANCELED
        order['RemainingQuantity'] = 0.0

@njit
def propagate_fill_up(orders, index, quantity):
    parent_index = orders[index]['ParentIndex']
    if parent_index != -1:
        parent_order = orders[parent_index]
        fill_quantity = min(parent_order['RemainingQuantity'], quantity)
        parent_order['FilledQuantity'] += fill_quantity
        parent_order['RemainingQuantity'] -= fill_quantity

        # Update state if fully filled
        if parent_order['RemainingQuantity'] == 0.0:
            parent_order['State'] = STATE_FILLED

        # Recursively propagate up
        propagate_fill_up(orders, parent_index, fill_quantity)

@njit
def propagate_unfill_up(orders, index, quantity):
    parent_index = orders[index]['ParentIndex']
    if parent_index != -1:
        parent_order = orders[parent_index]
        parent_order['FilledQuantity'] -= quantity
        parent_order['RemainingQuantity'] += quantity

        # Update state if necessary
        if parent_order['RemainingQuantity'] > 0.0 and parent_order['State'] == STATE_FILLED:
            parent_order['State'] = STATE_ACK

        # Recursively propagate up
        propagate_unfill_up(orders, parent_index, quantity)

@njit
def simulate_fill(orders, index, desired_quantity):
    order = orders[index]
    fill_quantity = min(order['RemainingQuantity'], desired_quantity)
    order['FilledQuantity'] += fill_quantity
    order['RemainingQuantity'] -= fill_quantity

    # Update state if fully filled
    if order['RemainingQuantity'] == 0.0:
        order['State'] = STATE_FILLED

    # Propagate fill up the hierarchy
    propagate_fill_up(orders, index, fill_quantity)

    return fill_quantity

@njit
def simulate_fills(orders, order_id_to_index, market_impact):
    for index in range(len(orders)):
        order = orders[index]

        if order['IsPing']:
            # Simulation logic specific to pings
            # Decide whether to fill pings based on internal logic
            desired_quantity = order['RemainingQuantity']
            fill_quantity = simulate_fill(orders, index, desired_quantity)

            # For internal pings, you may or may not update market impact
            # Here we assume pings do not impact the external market
        else:
            # Existing simulation logic for non-ping orders
            # Case 1: Fill an order that was not filled in reality
            if order['FilledQuantity'] == 0.0 and order['State'] == STATE_ACK:
                desired_quantity = order['RemainingQuantity']
                fill_quantity = simulate_fill(orders, index, desired_quantity)

                # Update market impact
                key = (order['StockSymbol'], order['Date'])
                if key in market_impact:
                    market_impact[key] += fill_quantity
                else:
                    market_impact[key] = fill_quantity

            # Case 2: Not fill an order which was filled in reality
            elif order['FilledQuantity'] > 0.0 and order['State'] == STATE_ACK:
                unfill_quantity = order['FilledQuantity']
                order['FilledQuantity'] -= unfill_quantity
                order['RemainingQuantity'] += unfill_quantity
                order['State'] = STATE_ACK  # Reset state if needed

                # Update market impact
                key = (order['StockSymbol'], order['Date'])
                if key in market_impact:
                    market_impact[key] -= unfill_quantity
                else:
                    market_impact[key] = -unfill_quantity

                # Propagate unfill up the hierarchy
                propagate_unfill_up(orders, index, unfill_quantity)

        # Ensure no overfilling
        if order['FilledQuantity'] > order['OriginalQuantity']:
            order['FilledQuantity'] = order['OriginalQuantity']
            order['RemainingQuantity'] = 0.0
            order['State'] = STATE_FILLED

@njit
def validate_orders(orders):
    for order in orders:
        if order['FilledQuantity'] > order['OriginalQuantity']:
            raise ValueError(f"Order {order['OrderId']} overfilled!")

# Function to print order details (for testing purposes)
def print_order_details(orders):
    print("Order Details:")
    print("Index | OrderId | ParentIndex | RootIndex | State | OrigQty | FilledQty | RemainQty | IsPing")
    for idx, order in enumerate(orders):
        print(f"{idx} | {order['OrderId']} | {order['ParentIndex']} | {order['RootIndex']} | {order['State']} | "
              f"{order['OriginalQuantity']} | {order['FilledQuantity']} | {order['RemainingQuantity']} | {order['IsPing']}")

# Function to print market impact (for testing purposes)
def print_market_impact(market_impact):
    print("\nMarket Impact:")
    for key in market_impact:
        stock_symbol, date = key
        impact = market_impact[key]
        print(f"StockSymbol: {stock_symbol}, Date: {date}, Market Impact: {impact}")

# Sample data generation for testing
def generate_sample_events(num_orders=1000):
    np.random.seed(42)  # For reproducibility
    events_list = []

    for i in range(num_orders):
        # Randomly decide if this is a ping or a regular order
        is_ping = np.random.rand() < 0.3  # 30% chance of being a ping
        event_type = EVENT_PING_ORDER if is_ping else EVENT_NEW_ORDER
        order_id = i + 1
        parent_order_id = np.random.randint(1, order_id) if order_id > 1 else -1
        quantity = np.random.uniform(10, 1000)
        timestamp = np.random.randint(1609459200, 1640995200)  # Random timestamp in 2021
        stock_symbol = np.random.randint(1, 100)  # Assume 100 different stocks
        date = int(timestamp // 86400)  # Simplified date representation
        events_list.append((event_type, order_id, parent_order_id, quantity, timestamp, stock_symbol, date, is_ping))

        # Randomly decide to add a fill or cancel event
        if np.random.rand() < 0.5:
            # Fill event
            fill_quantity = np.random.uniform(0, quantity)
            events_list.append((EVENT_FILL, order_id, -1, fill_quantity, timestamp + 100, stock_symbol, date, False))
        elif np.random.rand() < 0.2:
            # Cancel event
            events_list.append((EVENT_CANCEL, order_id, -1, 0.0, timestamp + 100, stock_symbol, date, False))

    return np.array(events_list, dtype=event_dtype)

# Main execution
def main():
    # Generate sample events
    events = generate_sample_events(num_orders=10000)  # Adjust the number for a reasonably sized dataset

    # Initialize data structures
    orders = List()
    order_id_to_index = Dict.empty(key_type=int64, value_type=int64)
    market_impact = Dict.empty(
        key_type=types.UniTuple(int64, 2),
        value_type=float64
    )

    # Process events and simulate fills
    process_events(events, orders, order_id_to_index, market_impact)
    simulate_fills(orders, order_id_to_index, market_impact)
    validate_orders(orders)

    # Print results
    print_order_details(orders[:20])  # Print first 20 orders for brevity
    print_market_impact(market_impact)

if __name__ == "__main__":
    main()
