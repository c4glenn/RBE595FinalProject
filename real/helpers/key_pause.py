"""Pause the robot from the keyboard while the task runs without asking (run_task.py).

Any key -> PAUSE: the move in progress is cancelled (the arm stops where it is: ArmMover._send),
and nothing else starts. Then:
    Enter / space / c   carry on (the stopped move is planned again from where the arm is)
    q                   stop the run (like Ctrl-C: the saved state is kept, `spice home` after)
Ctrl-C still works at any time (and cancels the move in progress too).

The terminal is put in cbreak mode (keys arrive one by one, no Enter needed, Ctrl-C still a
signal) while this runs, and put back by close(). Not a terminal (piped, ssh -T): does nothing.
"""
import contextlib
import os
import select
import sys
import termios
import threading
import tty

GO_KEYS = ('\n', '\r', ' ', 'c', 'C')
QUIT_KEYS = ('q', 'Q')


class KeyPause:
    def __init__(self, log=print, fd=None):
        self.log = log
        self.fd = sys.stdin.fileno() if fd is None else fd
        self._old = None
        self._thread = None
        self._on = False
        self._pause = threading.Event()
        self._go = threading.Event()
        self._quit = False

    # ------------------------------------------------------------------ terminal
    def start(self):
        """True if watching the keyboard (stdin is a terminal)."""
        if self._on:
            return True
        if not os.isatty(self.fd):
            return False
        self._old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self._on = True
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return True

    def close(self):
        """Stop watching and give the terminal back (input() works normally again)."""
        was = self._on
        self._on = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)
            self._old = None
        return was

    @contextlib.contextmanager
    def suspended(self):
        """For a real typed answer (input()) in the middle of a run."""
        was = self.close()
        try:
            yield
        finally:
            if was:
                self.start()

    def _read(self):
        while self._on:
            ready, _, _ = select.select([self.fd], [], [], 0.1)
            if not ready or not self._on:
                continue
            key = os.read(self.fd, 1).decode(errors='ignore')
            if not self._pause.is_set():
                self._go.clear()
                self._pause.set()
                self.log('\n*** PAUSE (key pressed): stopping the arm ***')
            elif key in GO_KEYS:
                self._go.set()
            elif key in QUIT_KEYS:
                self._quit = True
                self._go.set()

    # ------------------------------------------------------------------ for the task / the mover
    def requested(self):
        return self._pause.is_set()

    def hold(self):
        """If a pause was asked for: wait here until carry on (returns) or q (KeyboardInterrupt)."""
        if not self._pause.is_set():
            return
        self.log('PAUSED: the arm is stopped. Enter / space / c = carry on, q = stop the run.')
        while not self._go.wait(0.1):
            pass
        quit_, self._quit = self._quit, False
        self._go.clear()
        self._pause.clear()
        if quit_:
            raise KeyboardInterrupt('stopped from the keyboard (q)')
        self.log('carrying on')
