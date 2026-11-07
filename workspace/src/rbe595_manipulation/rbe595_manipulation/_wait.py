import threading


def wait_for_future(future, timeout_s: float) -> bool:
    """Block the calling thread until `future` completes or `timeout_s` elapses.

    Does NOT spin the node itself -- the node must already be spinning (e.g. under a
    MultiThreadedExecutor) on other threads, which is what actually services this future.
    Calling rclpy.spin_until_future_complete() instead would spin the node a second time from
    inside a callback that executor is already running, which is unsafe.
    """
    done_event = threading.Event()
    future.add_done_callback(lambda _: done_event.set())
    return done_event.wait(timeout_s)
