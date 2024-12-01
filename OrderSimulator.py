import numpy as np
from numba import njit, int64, float64, int8, boolean, types
from numba.experimental import jitclass
from numba.typed import List, Dict

# Constants for order states
STATE_PENDING = np.int8(0)
STATE_ACK = np.int8(1)
STATE_PENDING_CANCEL = np.int8(2)
STATE_CANCELED = np.int8(3)
STATE_FILLED = np.int8(4)

# Constants for event types
EVENT_NEW_ORDER = np.int8(0)
EVENT_FILL = np.int8(1)
EVENT_CANCEL = np.int8(2)
EVENT_PING_ORDER = np.int8(3)

# Define the Order class using jitclass
order_spec = [
    ('OrderId', int64),
    ('ParentIndex', int64),
    ('RootIndex', int64),
    ('State', int8),
    ('OriginalQuantity', float64),
    ('FilledQuantity', float64),
    ('RemainingQuantity', float64),
    ('StockSymbol', int64),
    ('Date', int64),
    ('IsPing', boolean)
]

@jitclass(order_spec)
class Order:
    def __init__(self, OrderId: int, ParentIndex: int, RootIndex: int, State: int,
                 OriginalQuantity: float, FilledQuantity: float, RemainingQuantity: float,
                 StockSymbol: int, Date: int, IsPing: bool):
        self.OrderId = OrderId
        self.ParentIndex = ParentIndex
        self.RootIndex = RootIndex
        self.State = State
        self.OriginalQuantity = OriginalQuantity
        self.FilledQuantity = FilledQuantity
        self.RemainingQuantity = RemainingQuantity
        self.StockSymbol = StockSymbol
        self.Date = Date
        self.IsPing = IsPing

# Event data structure
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
def process_events(events: np.ndarray, orders: List, order_id_to_index: Dict, market_impact: Dict):
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
def handle_new_order(event: np.ndarray, orders: List, order_id_to_index: Dict):
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
        root_index = orders[parent_index].RootIndex
    else:
        root_index = len(orders)  # Self root if no parent

    # Create new order
    order = Order(
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
def handle_fill(event: np.ndarray, orders: List, order_id_to_index: Dict, market_impact: Dict):
    order_id = event['OrderId']
    quantity = event['Quantity']

    if order_id in order_id_to_index:
        index = order_id_to_index[order_id]
        order = orders[index]

        # Calculate fill quantity
        fill_quantity = min(order.RemainingQuantity, quantity)
        order.FilledQuantity += fill_quantity
        order.RemainingQuantity -= fill_quantity

        # Update state if fully filled
        if order.RemainingQuantity <= 0.0:
            order.RemainingQuantity = 0.0
            order.State = STATE_FILLED

        # Update market impact
        key = (order.StockSymbol, order.Date)
        if key in market_impact:
            market_impact[key] += fill_quantity
        else:
            market_impact[key] = fill_quantity

        # Propagate fill up the hierarchy
        propagate_fill_up(orders, index, fill_quantity)

@njit
def handle_cancel(event: np.ndarray, orders: List, order_id_to_index: Dict):
    order_id = event['OrderId']

    if order_id in order_id_to_index:
        index = order_id_to_index[order_id]
        order = orders[index]
        order.State = STATE_CANCELED
        order.RemainingQuantity = 0.0

@njit
def calculate_max_fillable(orders: List, index: int, desired_quantity: float) -> float:
    order = orders[index]
    max_fillable = desired_quantity

    parent_index = order.ParentIndex
    while parent_index != -1:
        parent_order = orders[parent_index]
        available_to_fill = parent_order.OriginalQuantity - parent_order.FilledQuantity
        max_fillable = min(max_fillable, available_to_fill)
        parent_index = parent_order.ParentIndex

    return max_fillable

@njit
def simulate_fill(orders: List, index: int, desired_quantity: float) -> float:
    order = orders[index]

    # Determine the maximum fillable quantity without overfilling parents
    max_fillable = calculate_max_fillable(orders, index, desired_quantity)

    fill_quantity = min(order.RemainingQuantity, desired_quantity, max_fillable)
    order.FilledQuantity += fill_quantity
    order.RemainingQuantity -= fill_quantity

    # Update state if fully filled
    if order.RemainingQuantity <= 0.0:
        order.RemainingQuantity = 0.0
        order.State = STATE_FILLED

    # Propagate fill up the hierarchy
    propagate_fill_up(orders, index, fill_quantity)

    return fill_quantity

@njit
def propagate_fill_up(orders: List, index: int, quantity: float):
    parent_index = orders[index].ParentIndex
    if parent_index != -1:
        parent_order = orders[parent_index]

        # Calculate how much the parent can actually be filled without overfilling
        available_to_fill = parent_order.OriginalQuantity - parent_order.FilledQuantity
        fill_quantity = min(quantity, available_to_fill)

        parent_order.FilledQuantity += fill_quantity
        parent_order.RemainingQuantity = parent_order.OriginalQuantity - parent_order.FilledQuantity

        # Update state if fully filled
        if parent_order.RemainingQuantity <= 0.0:
            parent_order.RemainingQuantity = 0.0
            parent_order.State = STATE_FILLED

        # Recursively propagate up with the actual fill quantity
        propagate_fill_up(orders, parent_index, fill_quantity)

@njit
def propagate_unfill_up(orders: List, index: int, quantity: float):
    parent_index = orders[index].ParentIndex
    if parent_index != -1:
        parent_order = orders[parent_index]
        parent_order.FilledQuantity -= quantity
        parent_order.RemainingQuantity += quantity

        # Update state if necessary
        if parent_order.RemainingQuantity > 0.0 and parent_order.State == STATE_FILLED:
            parent_order.State = STATE_ACK

        # Recursively propagate up
        propagate_unfill_up(orders, parent_index, quantity)

@njit
def simulate_fills(orders: List, order_id_to_index: Dict, market_impact: Dict):
    for index in range(len(orders)):
        order = orders[index]

        if order.IsPing:
            # Simulation logic specific to pings
            desired_quantity = order.RemainingQuantity
            fill_quantity = simulate_fill(orders, index, desired_quantity)
            # Assuming pings do not impact the external market
        else:
            # Existing simulation logic for non-ping orders
            # Case 1: Fill an order that was not filled in reality
            if order.FilledQuantity == 0.0 and order.State == STATE_ACK:
                desired_quantity = order.RemainingQuantity
                fill_quantity = simulate_fill(orders, index, desired_quantity)

                # Update market impact
                key = (order.StockSymbol, order.Date)
                if key in market_impact:
                    market_impact[key] += fill_quantity
                else:
                    market_impact[key] = fill_quantity

            # Case 2: Not fill an order which was filled in reality
            elif order.FilledQuantity > 0.0 and order.State == STATE_ACK:
                unfill_quantity = order.FilledQuantity
                order.FilledQuantity -= unfill_quantity
                order.RemainingQuantity += unfill_quantity
                order.State = STATE_ACK  # Reset state if needed

                # Update market impact
                key = (order.StockSymbol, order.Date)
                if key in market_impact:
                    market_impact[key] -= unfill_quantity
                else:
                    market_impact[key] = -unfill_quantity

                # Propagate unfill up the hierarchy
                propagate_unfill_up(orders, index, unfill_quantity)

        # Ensure no overfilling
        if order.FilledQuantity > order.OriginalQuantity:
            order.FilledQuantity = order.OriginalQuantity
            order.RemainingQuantity = 0.0
            order.State = STATE_FILLED

@njit
def validate_orders(orders: List):
    for order in orders:
        if order.FilledQuantity > order.OriginalQuantity + 1e-6:
            # Allowing a small epsilon due to floating-point precision
            raise ValueError(f"Order {order.OrderId} overfilled!")

# Function to print order details (for testing purposes)
def print_order_details(orders: List):
    print("Order Details:")
    print("Index | OrderId | ParentIndex | RootIndex | State | OrigQty | FilledQty | RemainQty | IsPing")
    for idx, order in enumerate(orders):
        print(f"{idx} | {order.OrderId} | {order.ParentIndex} | {order.RootIndex} | {order.State} | "
              f"{order.OriginalQuantity:.2f} | {order.FilledQuantity:.2f} | {order.RemainingQuantity:.2f} | {order.IsPing}")

# Function to print market impact (for testing purposes)
def print_market_impact(market_impact: Dict):
    print("\nMarket Impact:")
    for key in market_impact:
        stock_symbol, date = key
        impact = market_impact[key]
        print(f"StockSymbol: {stock_symbol}, Date: {date}, Market Impact: {impact:.2f}")

# Sample data generation for testing
def generate_sample_events(num_orders: int = 1000) -> np.ndarray:
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
        rand_val = np.random.rand()
        if rand_val < 0.5:
            # Fill event
            fill_quantity = np.random.uniform(0, quantity)
            events_list.append((EVENT_FILL, order_id, -1, fill_quantity, timestamp + 100, stock_symbol, date, False))
        elif rand_val < 0.7:
            # Cancel event
            events_list.append((EVENT_CANCEL, order_id, -1, 0.0, timestamp + 100, stock_symbol, date, False))

    return np.array(events_list, dtype=event_dtype)

# Main execution
def main():
    # Generate sample events
    events = generate_sample_events(num_orders=10000)  # Adjust the number for a reasonably sized dataset

    # Initialize data structures
    orders = List.empty_list(Order.class_type.instance_type)
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
