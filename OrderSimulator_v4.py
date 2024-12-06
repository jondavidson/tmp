import numpy as np
from numba import njit, jitclass, types
from numba.typed import List, Dict

# States
STATE_NEW = 0
STATE_ACK = 1
STATE_REJECTED = 2
STATE_PENDING_CANCEL = 3
STATE_CANCELED = 4
STATE_FILLED = 5
STATE_PART_FILLED = 6
STATE_PENDING_REPLACE = 7

# Event Types
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

order_spec = [
    ('OrderId', types.int64),
    ('ParentId', types.int64),
    ('RootId', types.int64),
    ('State', types.int64),
    ('OriginalQuantity', types.int64),
    ('FilledQuantity', types.int64),
    ('RemainingQuantity', types.int64),
    ('StockSymbol', types.int64),
    ('Date', types.int64),
    ('IsPing', types.boolean),
    ('RequestedNewQuantity', types.int64),
    ('HasRequestedNewQuantity', types.boolean),
    ('PreviousState', types.int64)
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

@njit
def propagate_up_down(orders, order_id_to_index, parent_to_children, idx):
    """
    After each event, ensure the parent's fill and remaining quantities are consistent
    with the children's aggregated fills. Also handle parent replace scenarios.

    Upward propagation:
    - Recalculate parent's FilledQuantity = sum of children's fills + parent's own fill (if any)
      (In this scenario, the parent does not have a separate 'direct fill' beyond children, 
       but if it did, we’d add it here.)
    - RemainingQuantity = OriginalQuantity - FilledQuantity (clamped at 0)
    - State adjustments based on fill ratio.

    If parent's FilledQuantity > parent's OriginalQuantity after a replace that lowered the parent's quantity,
    we cap FilledQuantity at OriginalQuantity, set RemainingQuantity=0, and mark it FILLED.

    This logic cascades upward until we reach the root or a parent with no parent.
    """
    current_idx = idx
    while True:
        o = orders[current_idx]
        parent_id = o.ParentId
        if parent_id == -1:
            # Reached top
            break
        if parent_id not in order_id_to_index:
            # Parent not found (should not happen if data is consistent)
            break
        parent_idx = order_id_to_index[parent_id]
        parent = orders[parent_idx]

        # Aggregate children fills
        total_filled = 0
        if parent_id in parent_to_children:
            children_ids = parent_to_children[parent_id]
            for c_id in children_ids:
                c_idx = order_id_to_index[c_id]
                c = orders[c_idx]
                total_filled += c.FilledQuantity

        # Update parent's fill and remain
        parent.FilledQuantity = total_filled
        if parent.FilledQuantity > parent.OriginalQuantity:
            # Logical inconsistency, but handle gracefully:
            parent.FilledQuantity = parent.OriginalQuantity
            parent.RemainingQuantity = 0
            parent.State = STATE_FILLED
        else:
            parent.RemainingQuantity = parent.OriginalQuantity - parent.FilledQuantity
            # Update state based on fills
            if parent.FilledQuantity == parent.OriginalQuantity:
                parent.State = STATE_FILLED
            elif parent.FilledQuantity > 0 and parent.FilledQuantity < parent.OriginalQuantity:
                # If partially filled, ensure not canceled or rejected
                if parent.State not in (STATE_CANCELED, STATE_FILLED, STATE_REJECTED):
                    parent.State = STATE_PART_FILLED
            else:
                # No fills:
                # If parent was ACK and no fill: remain ACK
                # If parent was NEW and never ACKed: remain NEW
                # If parent was PART_FILLED and now no fill (unlikely): remain PART_FILLED if previously so.
                # Generally do not override parent's state unless fill conditions dictate.
                pass

        current_idx = parent_idx

@njit
def process_events(events, orders, order_id_to_index, parent_to_children):
    for i in range(events.shape[0]):
        etype = events[i]['EventType']
        oid = events[i]['OrderId']
        poid = events[i]['ParentOrderId']
        qty = events[i]['Quantity']
        stock_symbol = events[i]['StockSymbol']
        date = events[i]['Date']
        is_ping = (events[i]['IsPing'] != 0)

        event_affects_idx = -1

        if etype == EVENT_ORDER_NEW:
            # Create a new order
            if poid in order_id_to_index:
                parent_idx = order_id_to_index[poid]
                root_id = orders[parent_idx].RootId
            else:
                parent_idx = -1
                root_id = len(orders)
            o = Order(oid, poid, root_id, STATE_NEW, qty, 0, qty, stock_symbol, date, is_ping)
            orders.append(o)
            order_id_to_index[oid] = len(orders)-1
            event_affects_idx = len(orders)-1

            # Register child
            if poid != -1:
                if poid not in parent_to_children:
                    parent_to_children[poid] = List.empty_list(types.int64)
                children_list = parent_to_children[poid]
                children_list.append(oid)
        
        else:
            # Existing order
            if oid not in order_id_to_index:
                continue
            idx = order_id_to_index[oid]
            o = orders[idx]

            if etype == EVENT_ORDER_ACK:
                if o.State == STATE_NEW:
                    o.State = STATE_ACK
                    event_affects_idx = idx

            elif etype == EVENT_ORDER_REJECT:
                if o.State == STATE_NEW:
                    o.State = STATE_REJECTED
                    event_affects_idx = idx

            elif etype == EVENT_FILL:
                # Fill if ACK or PART_FILLED
                if o.State in (STATE_ACK, STATE_PART_FILLED):
                    fill_qty = qty
                    if fill_qty > o.RemainingQuantity:
                        fill_qty = o.RemainingQuantity
                    o.FilledQuantity += fill_qty
                    o.RemainingQuantity -= fill_qty
                    if o.RemainingQuantity == 0:
                        o.State = STATE_FILLED
                    else:
                        if o.State not in (STATE_CANCELED, STATE_FILLED, STATE_REJECTED):
                            o.State = STATE_PART_FILLED
                    event_affects_idx = idx

            elif etype == EVENT_CANCEL_NEW:
                # Cancel if ACK, PART_FILLED, or NEW
                if o.State in (STATE_ACK, STATE_PART_FILLED, STATE_NEW):
                    o.PreviousState = o.State
                    o.State = STATE_PENDING_CANCEL
                    event_affects_idx = idx

            elif etype == EVENT_CANCEL_ACK:
                if o.State == STATE_PENDING_CANCEL:
                    o.State = STATE_CANCELED
                    o.PreviousState = -1
                    event_affects_idx = idx

            elif etype == EVENT_CANCEL_REJECT:
                if o.State == STATE_PENDING_CANCEL and o.PreviousState != -1:
                    o.State = o.PreviousState
                    o.PreviousState = -1
                    event_affects_idx = idx

            elif etype == EVENT_REPLACE_NEW:
                # Replace if ACK or PART_FILLED
                if o.State in (STATE_ACK, STATE_PART_FILLED):
                    o.PreviousState = o.State
                    o.RequestedNewQuantity = qty
                    o.HasRequestedNewQuantity = True
                    o.State = STATE_PENDING_REPLACE
                    event_affects_idx = idx

            elif etype == EVENT_REPLACE_ACK:
                if o.State == STATE_PENDING_REPLACE and o.HasRequestedNewQuantity:
                    new_qty = o.RequestedNewQuantity
                    old_filled = o.FilledQuantity
                    o.OriginalQuantity = new_qty
                    # Adjust remain
                    if old_filled > o.OriginalQuantity:
                        # If already overfilled compared to new original, cap it
                        o.FilledQuantity = o.OriginalQuantity
                        o.RemainingQuantity = 0
                        o.State = STATE_FILLED
                    else:
                        o.RemainingQuantity = o.OriginalQuantity - o.FilledQuantity
                        if o.RemainingQuantity == 0:
                            o.State = STATE_FILLED
                        else:
                            # If old_filled >0 and less than original
                            if old_filled > 0:
                                o.State = STATE_PART_FILLED
                            else:
                                # No fills yet, revert to ACK
                                o.State = STATE_ACK
                    o.HasRequestedNewQuantity = False
                    o.RequestedNewQuantity = 0
                    o.PreviousState = -1
                    event_affects_idx = idx

                # If not HasRequestedNewQuantity or wrong state, ignore

            elif etype == EVENT_REPLACE_REJECT:
                if o.State == STATE_PENDING_REPLACE:
                    if o.PreviousState != -1:
                        o.State = o.PreviousState
                        o.PreviousState = -1
                    o.HasRequestedNewQuantity = False
                    o.RequestedNewQuantity = 0
                    event_affects_idx = idx

        # After updating the order, propagate changes up
        if event_affects_idx != -1:
            propagate_up_down(orders, order_id_to_index, parent_to_children, event_affects_idx)

@njit
def print_orders_jit(orders):
    n = len(orders)
    result = np.empty(n, dtype=np.dtype([
        ('OrderId', np.int64),
        ('ParentId', np.int64),
        ('RootId', np.int64),
        ('State', np.int64),
        ('OriginalQuantity', np.int64),
        ('FilledQuantity', np.int64),
        ('RemainingQuantity', np.int64),
        ('IsPing', np.bool_)
    ]))
    for i in range(n):
        o = orders[i]
        result[i]['OrderId'] = o.OrderId
        result[i]['ParentId'] = o.ParentId
        result[i]['RootId'] = o.RootId
        result[i]['State'] = o.State
        result[i]['OriginalQuantity'] = o.OriginalQuantity
        result[i]['FilledQuantity'] = o.FilledQuantity
        result[i]['RemainingQuantity'] = o.RemainingQuantity
        result[i]['IsPing'] = o.IsPing
    return result

def print_orders(orders):
    arr = print_orders_jit(orders)
    for i in range(arr.shape[0]):
        print(f"Index={i}, "
              f"OrderId={arr[i]['OrderId']}, "
              f"ParentId={arr[i]['ParentId']}, "
              f"RootId={arr[i]['RootId']}, "
              f"State={arr[i]['State']}, "
              f"Orig={arr[i]['OriginalQuantity']}, "
              f"Filled={arr[i]['FilledQuantity']}, "
              f"Rem={arr[i]['RemainingQuantity']}, "
              f"IsPing={arr[i]['IsPing']}")

def run_test(test_name: str, events):
    print(f"\n--- Running Test: {test_name} ---")
    from numba.typed import List, Dict
    orders = List.empty_list(Order)
    order_id_to_index = Dict.empty(key_type=types.int64, value_type=types.int64)
    parent_to_children = Dict.empty(key_type=types.int64, value_type=types.ListType(types.int64))
    process_events(events, orders, order_id_to_index, parent_to_children)
    print("Final Orders State:")
    print_orders(orders)

def main():
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

    # Example complex scenario:
    # O1: Large root order
    # O2, O3: children under O1
    # O4 under O2
    # We do fills, cancels, replaces at multiple levels to test aggregator logic.

    events_data = np.array([
        (EVENT_ORDER_NEW,    1, -1, 10000,    0,   100, 20210101, 0),
        (EVENT_ORDER_ACK,    1, -1, 0,         1,   100, 20210101, 0),
        (EVENT_FILL,         1, -1, 2000,      2,   100, 20210101, 0),

        (EVENT_ORDER_NEW,    2, 1,  4000,      3,   100, 20210101, 0),
        (EVENT_ORDER_ACK,    2, 1,  0,         4,   100, 20210101, 0),
        (EVENT_FILL,         2, 1,  1000,       5,   100, 20210101, 0),

        (EVENT_ORDER_NEW,    3, 1,  2000,       6,   100, 20210101, 1),
        (EVENT_ORDER_ACK,    3, 1,  0,          7,   100, 20210101, 1),
        (EVENT_CANCEL_NEW,   3, 1,  0,          8,   100, 20210101, 1),
        (EVENT_CANCEL_ACK,   3, 1,  0,          9,   100, 20210101, 1),

        (EVENT_REPLACE_NEW,  2, 1,  2000,       10,  100, 20210101, 0),
        (EVENT_REPLACE_ACK,  2, 1,  0,          11,  100, 20210101, 0),

        (EVENT_ORDER_NEW,    4, 2,  500,        12,  100, 20210101, 0),
        (EVENT_ORDER_ACK,    4, 2,  0,          13,  100, 20210101, 0),
        (EVENT_FILL,         4, 2,  500,        14,  100, 20210101, 0),

        # Attempt to cancel a filled O4
        (EVENT_CANCEL_NEW,   4, 2,  0,          15,  100, 20210101, 0),
        (EVENT_CANCEL_REJECT,4, 2,  0,          16,  100, 20210101, 0),

        # Replace O1 to a larger quantity
        (EVENT_REPLACE_NEW,  1, -1, 12000,      17,  100, 20210101, 0),
        (EVENT_REPLACE_ACK,  1, -1, 0,          18,  100, 20210101, 0),

        # Additional tests could be inserted here (e.g., reduce O1 below filled child's quantity)
        # but we have established how we handle such scenarios.
    ], dtype=event_dtype)

    run_test("Complex Scenario with Full Logic & Aggregation", events_data)

if __name__ == "__main__":
    main()
