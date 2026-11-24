"""Keyboard pause (helpers/key_pause.py) on a pseudo-terminal, and ArmMover's cancel + plan-again
on a pause / Ctrl-C with a fake action goal (no robot; the ArmMover part needs ROS sourced).

  python3 real/test/test_key_pause.py
"""
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from helpers.key_pause import KeyPause                         # noqa: E402


def keys():
    master, slave = os.openpty()
    log = []
    kp = KeyPause(log=log.append, fd=slave)
    assert kp.start()
    return kp, master, log


def wait(cond, t=2.0):
    end = time.time() + t
    while not cond():
        assert time.time() < end, 'timed out'
        time.sleep(0.01)


def test_pause_and_go():
    kp, master, log = keys()
    assert not kp.requested()
    kp.hold()                                     # nothing pressed: returns at once
    os.write(master, b'x')
    wait(kp.requested)
    threading.Timer(0.3, lambda: os.write(master, b'\n')).start()
    t0 = time.time()
    kp.hold()                                     # waits for Enter
    assert time.time() - t0 > 0.2 and not kp.requested()
    os.write(master, b' ')                        # another key: pause again, q: stop
    wait(kp.requested)
    os.write(master, b'q')
    try:
        kp.hold()
        raise AssertionError('q did not stop')
    except KeyboardInterrupt:
        pass
    with kp.suspended():                          # terminal back to normal for a typed answer
        assert not kp._on
    assert kp._on
    kp.close()
    print('key pause ok')


class Future:
    def __init__(self, value=None, done=False):
        self.value, self._done = value, done

    def done(self):
        return self._done

    def result(self):
        return self.value

    def finish(self, value):
        self.value, self._done = value, True


class Goal:
    """A move that takes `duration` s unless cancelled."""
    def __init__(self, duration):
        self.accepted = True
        self.cancelled = False
        self.result_future = Future()
        threading.Timer(duration, self._end).start()

    def _end(self):
        if not self.cancelled:
            self.result_future.finish(type('R', (), {'result': 'moved'})())

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancelled = True
        self.result_future.finish(type('R', (), {'result': 'cancelled'})())
        return Future(done=True)


class Client:
    def __init__(self, durations):
        self.durations, self.goals = list(durations), []

    def send_goal_async(self, goal):
        g = Goal(self.durations.pop(0))
        self.goals.append(g)
        return Future(g, done=True)


def test_mover_cancel_and_again():
    try:
        from helpers.robot_motion import ArmMover, again_after_pause
    except ImportError as e:
        print(f'mover part skipped (ROS not sourced: {e})')
        return

    class Log:
        def __getattr__(self, name):
            return lambda *a: None

    class Mover(ArmMover):
        def __init__(self, client):
            self.client, self.log, self.pause = client, Log(), None

        @again_after_pause
        def move(self):
            return self._send(self.client, 'goal')

    kp, master, _ = keys()
    m = Mover(Client([1.0, 0.2]))
    m.pause = kp
    threading.Timer(0.2, lambda: os.write(master, b'k')).start()        # pause part way
    threading.Timer(0.6, lambda: os.write(master, b'c')).start()        # carry on
    assert m.move() == 'moved'
    assert len(m.client.goals) == 2 and m.client.goals[0].cancelled and not m.client.goals[1].cancelled
    print('mover: paused move cancelled, planned again after carry on ok')

    m = Mover(Client([1.0]))
    m.pause = kp
    threading.Timer(0.2, lambda: os.write(master, b'k')).start()
    threading.Timer(0.5, lambda: os.write(master, b'q')).start()        # stop
    try:
        m.move()
        raise AssertionError('q did not stop the move')
    except KeyboardInterrupt:
        pass
    assert m.client.goals[0].cancelled
    print('mover: q cancels and stops ok')
    kp.close()


if __name__ == '__main__':
    test_pause_and_go()
    test_mover_cancel_and_again()
    print('\nALL OK')
