import numpy as np
from numba import njit, int64, int8, boolean, types
from numba.experimental import jitclass
from numba.typed import List, Dict

# -----------------------------------
# Constants for States (int8)
# -----------------------------------
STATE_NEW = 0
STATE_ACK = 1
STATE_REJECTED = 2
STATE_PENDING_CANCEL = 3
STATE_CANCELED = 4
STATE_FILLED = 5
STATE_PART_FILLED = 6
STATE_PENDING_REPLACE = 7

# -----------------------------------
# Constants for Events (int8)
# -----------------------------------
EVENT_ORDER_NEW = 0
EVENT_ORDER_ACK = 1
EVENT_ORDER_REJECT = 2
EVENT_CANCEL_NEW = 3
EVENT_CANCEL_ACK = 4
EVENT_CANCEL_REJECT = 5
EVENT_REPLACE_NEW = 6
EVENT_REPLACE_ACK = 7
EVENT_REPLACE_REJECT = 8
EVENT_FILL = 9

# -----------------------------------
# Order jitclass Definition
# -----------------------------------
order_spec = [
    ('OrderId', int64),
    ('ParentId', int64),
    ('RootId', int64),
    ('State', int8),
    ('OriginalQuantity', int64),
    ('FilledQuantity', int64),
    ('RemainingQuantity', int64),
    ('StockSymbol', int64),
    ('Date', int64),
    ('IsPing', boolean),

    ('RequestedNewQuantity', int64),     # For replace requests
    ('HasRequestedNewQuantity', boolean),# Whether a new quantity was requested
    ('PreviousState', int8)              # For revert on reject
]

@jitclass(order_spec)
class Order:
    def __init__(self, OrderId, ParentId, RootId, State,
                 OriginalQuantity, FilledQuantity, RemainingQuantity,
                 StockSymbol, Date, IsPing):
        self.OrderId = OrderId
        self.ParentId = ParentId
        self.RootId = RootId
        self.State = State
        self.OriginalQuantity = OriginalQuantity
        self.FilledQuantity = FilledQuantity
        self.RemainingQuantity = RemainingQuantity
        self.StockSymbol = StockSymbol
        self.Date = Date
        self.IsPing = IsPing
        self.RequestedNewQuantity = 0
        self.HasRequestedNewQuantity = False
        self.PreviousState = -1

# -----------------------------------
# Helper Functions
# -----------------------------------

@njit
def calculate_max_fillable(orders, index, desired_quantity):
    # type: (List[Order], int, int) -> int
    order = orders[index]
    max_fillable = desired_quantity
    parent_index = order.ParentId
    while parent_index != -1:
        parent_order = orders[parent_index]
        available_to_fill = parent_order.OriginalQuantity - parent_order.FilledQuantity
        if available_to_fill < max_fillable:
            max_fillable = available_to_fill
        parent_index = parent_order.ParentId
    return max_fillable

@njit
def propagate_fill_up(orders, index, quantity):
    # type: (List[Order], int, int) -> None
    if quantity <= 0:
        return
    parent_index = orders[index].ParentId
    if parent_index != -1:
        parent_order = orders[parent_index]
        available_to_fill = parent_order.OriginalQuantity - parent_order.FilledQuantity
        fill_quantity = min(quantity, available_to_fill)

        parent_order.FilledQuantity += fill_quantity
        parent_order.RemainingQuantity = parent_order.OriginalQuantity - parent_order.FilledQuantity
        if parent_order.RemainingQuantity == 0:
            parent_order.State = STATE_FILLED
        elif parent_order.State == STATE_ACK and parent_order.FilledQuantity > 0 and parent_order.RemainingQuantity > 0:
            parent_order.State = STATE_PART_FILLED

        propagate_fill_up(orders, parent_index, fill_quantity)

@njit
def propagate_unfill_up(orders, index, quantity):
    # type: (List[Order], int, int) -> None
    # If we need to revert fills, for example on replace reject or cancel reject
    parent_index = orders[index].ParentId
    if parent_index != -1 and quantity > 0:
        parent_order = orders[parent_index]
        parent_order.FilledQuantity -= quantity
        parent_order.RemainingQuantity = parent_order.OriginalQuantity - parent_order.FilledQuantity

        # Update state if needed
        if parent_order.State == STATE_FILLED and parent_order.RemainingQuantity > 0:
            # Revert to ACK or PART_FILLED
            # For simplicity, assume ACK if previously fully acked
            # If we had PART_FILLED before, we might guess:
            # If partial fill still occurs, keep PART_FILLED, else ACK
            if parent_order.FilledQuantity > 0:
                parent_order.State = STATE_PART_FILLED
            else:
                parent_order.State = STATE_ACK

        propagate_unfill_up(orders, parent_index, quantity)

# -----------------------------------
# Event Handlers
# -----------------------------------

@njit
def handle_order_new(event, orders, order_id_to_index):
    # type: (np.ndarray, List[Order], Dict[int64,int64]) -> None
    order_id = event[1]
    parent_order_id = event[2]
    quantity = event[3]
    timestamp = event[4]
    stock_symbol = event[5]
    date = event[6]
    is_ping = event[7] != 0

    # Find parent index if any
    parent_index = -1
    if parent_order_id in order_id_to_index:
        parent_index = order_id_to_index[parent_order_id]
        root_index = orders[parent_index].RootId
    else:
        root_index = len(orders)

    # Create order
    o = Order(
        OrderId=order_id,
        ParentId=parent_index,
        RootId=root_index,
        State=STATE_NEW,
        OriginalQuantity=quantity,
        FilledQuantity=0,
        RemainingQuantity=quantity,
        StockSymbol=stock_symbol,
        Date=date,
        IsPing=is_ping
    )
    orders.append(o)
    order_id_to_index[order_id] = len(orders) - 1

@njit
def handle_order_ack(event, orders, order_id_to_index):
    order_id = event[1]
    if order_id in order_id_to_index:
        idx = order_id_to_index[order_id]
        order = orders[idx]
        if order.State == STATE_NEW:
            order.State = STATE_ACK

@njit
def handle_order_reject(event, orders, order_id_to_index):
    order_id = event[1]
    if order_id in order_id_to_index:
        idx = order_id_to_index[order_id]
        order = orders[idx]
        # Reject if in NEW or pending states
        if order.State in (STATE_NEW, STATE_PENDING_CANCEL, STATE_PENDING_REPLACE):
            # If was pending replace or pending cancel, revert
            # Here just set REJECTED
            order.State = STATE_REJECTED
            order.RequestedNewQuantity = 0
            order.HasRequestedNewQuantity = False
            order.PreviousState = -1

@njit
def handle_cancel_new(event, orders, order_id_to_index):
    order_id = event[1]
    if order_id in order_id_to_index:
        idx = order_id_to_index[order_id]
        order = orders[idx]
        if order.State in (STATE_ACK, STATE_PART_FILLED):
            order.PreviousState = order.State
            order.State = STATE_PENDING_CANCEL

@njit
def handle_cancel_ack(event, orders, order_id_to_index):
    order_id = event[1]
    if order_id in order_id_to_index:
        idx = order_id_to_index[order_id]
        order = orders[idx]
        if order.State == STATE_PENDING_CANCEL:
            order.State = STATE_CANCELED
            order.RemainingQuantity = 0

@njit
def handle_cancel_reject(event, orders, order_id_to_index):
    order_id = event[1]
    if order_id in order_id_to_index:
        idx = order_id_to_index[order_id]
        order = orders[idx]
        if order.State == STATE_PENDING_CANCEL:
            # revert to previous state
            if order.PreviousState != -1:
                order.State = order.PreviousState
                order.PreviousState = -1

@njit
def handle_replace_new(event, orders, order_id_to_index):
    order_id = event[1]
    new_quantity = event[3]
    if order_id in order_id_to_index:
        idx = order_id_to_index[order_id]
        order = orders[idx]
        if order.State in (STATE_ACK, STATE_PART_FILLED):
            order.PreviousState = order.State
            order.State = STATE_PENDING_REPLACE
            order.RequestedNewQuantity = new_quantity
            order.HasRequestedNewQuantity = True

@njit
def handle_replace_ack(event, orders, order_id_to_index):
    order_id = event[1]
    if order_id in order_id_to_index:
        idx = order_id_to_index[order_id]
        order = orders[idx]
        if order.State == STATE_PENDING_REPLACE and order.HasRequestedNewQuantity:
            old_original = order.OriginalQuantity
            new_original = order.RequestedNewQuantity
            delta = new_original - old_original
            # Adjust quantities
            order.OriginalQuantity = new_original
            order.RemainingQuantity += delta
            # Return to previous state or ACK if unsure
            order.State = STATE_ACK
            order.PreviousState = -1
            order.RequestedNewQuantity = 0
            order.HasRequestedNewQuantity = False

@njit
def handle_replace_reject(event, orders, order_id_to_index):
    order_id = event[1]
    if order_id in order_id_to_index:
        idx = order_id_to_index[order_id]
        order = orders[idx]
        if order.State == STATE_PENDING_REPLACE:
            # revert to previous state
            if order.PreviousState != -1:
                order.State = order.PreviousState
                order.PreviousState = -1
            order.RequestedNewQuantity = 0
            order.HasRequestedNewQuantity = False

@njit
def handle_fill(event, orders, order_id_to_index):
    order_id = event[1]
    fill_quantity = event[3]
    if order_id in order_id_to_index:
        idx = order_id_to_index[order_id]
        order = orders[idx]
        # Only fill if ACK, PART_FILLED or PENDING_REPLACE (?), but let's allow fill if ACK or PART_FILLED
        # Also ensure we don't overfill
        if order.State in (STATE_ACK, STATE_PART_FILLED, STATE_PENDING_REPLACE):
            max_fillable = calculate_max_fillable(orders, idx, fill_quantity)
            actual_fill = min(order.RemainingQuantity, fill_quantity, max_fillable)
            order.FilledQuantity += actual_fill
            order.RemainingQuantity -= actual_fill
            if order.RemainingQuantity == 0:
                order.State = STATE_FILLED
            else:
                if order.FilledQuantity > 0 and order.State == STATE_ACK:
                    order.State = STATE_PART_FILLED
            # Propagate fill up
            propagate_fill_up(orders, idx, actual_fill)

# -----------------------------------
# Process Events Function
# -----------------------------------
# Event structure: (EventType, OrderId, ParentOrderId, Quantity, Timestamp, StockSymbol, Date, IsPing)
# All ints, IsPing = 0 or 1

@njit
def process_events(events, orders, order_id_to_index):
    # type: (np.ndarray, List[Order], Dict[int64,int64]) -> None
    for i in range(events.shape[0]):
        e = events[i]
        event_type = e[0]
        if event_type == EVENT_ORDER_NEW:
            handle_order_new(e, orders, order_id_to_index)
        elif event_type == EVENT_ORDER_ACK:
            handle_order_ack(e, orders, order_id_to_index)
        elif event_type == EVENT_ORDER_REJECT:
            handle_order_reject(e, orders, order_id_to_index)
        elif event_type == EVENT_CANCEL_NEW:
            handle_cancel_new(e, orders, order_id_to_index)
        elif event_type == EVENT_CANCEL_ACK:
            handle_cancel_ack(e, orders, order_id_to_index)
        elif event_type == EVENT_CANCEL_REJECT:
            handle_cancel_reject(e, orders, order_id_to_index)
        elif event_type == EVENT_REPLACE_NEW:
            handle_replace_new(e, orders, order_id_to_index)
        elif event_type == EVENT_REPLACE_ACK:
            handle_replace_ack(e, orders, order_id_to_index)
        elif event_type == EVENT_REPLACE_REJECT:
            handle_replace_reject(e, orders, order_id_to_index)
        elif event_type == EVENT_FILL:
            handle_fill(e, orders, order_id_to_index)

# -----------------------------------
# Example Usage
# -----------------------------------
def main():
    # Prepare typed structures
    orders = List.empty_list(Order.class_type.instance_type)
    order_id_to_index = Dict.empty(key_type=types.int64, value_type=types.int64)

    # Create sample events array
    # dtype: (EventType, OrderId, ParentOrderId, Quantity, Timestamp, StockSymbol, Date, IsPing)
    event_dtype = np.dtype([
        ('EventType', np.int8),
        ('OrderId', np.int64),
        ('ParentOrderId', np.int64),
        ('Quantity', np.int64),
        ('Timestamp', np.int64),
        ('StockSymbol', np.int64),
        ('Date', np.int64),
        ('IsPing', np.int8)
    ])

    events_data = np.array([
        (EVENT_ORDER_NEW, 1, -1, 100, 0, 10, 20210101, 0),
        (EVENT_ORDER_ACK, 1, -1, 0, 1, 10, 20210101, 0),
        (EVENT_FILL, 1, -1, 50, 2, 10, 20210101, 0),
        (EVENT_REPLACE_NEW, 1, -1, 120, 3, 10, 20210101, 0),
        (EVENT_REPLACE_ACK, 1, -1, 0, 4, 10, 20210101, 0),
        (EVENT_FILL, 1, -1, 70, 5, 10, 20210101, 0),
        (EVENT_CANCEL_NEW, 1, -1, 0, 6, 10, 20210101, 0),
        (EVENT_CANCEL_REJECT, 1, -1, 0, 7, 10, 20210101, 0),
        (EVENT_CANCEL_NEW, 1, -1, 0, 8, 10, 20210101, 0),
        (EVENT_CANCEL_ACK, 1, -1, 0, 9, 10, 20210101, 0)
    ], dtype=event_dtype)

    process_events(events_data, orders, order_id_to_index)

    # Print results
    for idx in range(len(orders)):
        o = orders[idx]
        print(f"OrderId={o.OrderId}, State={o.State}, Orig={o.OriginalQuantity}, "
              f"Filled={o.FilledQuantity}, Rem={o.RemainingQuantity}")

if __name__ == "__main__":
    main()
