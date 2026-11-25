#!/usr/bin/env python3
"""Recipe window: first the arm looks at the rack (run_task.py --scan: survey pose, every lid's
label read) and lists the spices that are there; pick them, in order; Start runs the whole task
(steps 1-8, run_task.py --spice NAME) once per spice, one after the other. Rescan after changing
the jars.

  real/spice.sh recipe            (starts robot + camera, then this window)
  python3 real/tools/recipe_gui.py   (robot + camera already running: NO_BRINGUP=1 spice bringup)

Runs are fully automatic: a run that goes well asks nothing (the jar is let go in the pockets
without asking). The only questions come after a failure ("the jar is in the hand: let go of it
here before going home?", "bring the arm home?" when not --home); they are answered No
automatically (carry the jar home, never drop it where the arm stopped), unless "debug: answer
the robot's questions myself" is ticked: then they show up with Yes / No buttons.
Every run goes in a pseudo-terminal, so Pause / Continue / Stop send the keys the run listens to
(any key = pause, Enter = carry on, q = stop). A failed run stops the recipe (with --home the arm
goes home); Start again carries on from the first spice not done.
Amounts (grams) are not in yet: one pour per spice.
"""
import argparse
import json
import os
import pty
import queue
import re
import select
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
ANSI = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')
MODES = {'shake (steps 1-8)': [],
         'pour test (find the pour angle)': ['--pour-test'],
         'no pouring (steps 1-4, 6-8)': ['--steps', '1-4,6-8']}


class RecipeApp:
    def __init__(self, root, cfg, scan=True):
        self.root, self.cfg = root, cfg
        self.spices = list((cfg.get('labels') or {}).get('spices') or ['salt'])
        self.job = None              # what the running process is: 'scan' or 'spice'
        self.scan_text = ''          # the scan's output so far (its SCAN_RESULT line)
        self.proc = None
        self.master = None
        self.out = queue.Queue()
        self.running = False
        self.current = None          # index in the recipe of the spice being run
        self.status = []             # per recipe line: '', 'running', 'done', 'FAILED'
        self.prompt = False

        root.title('Spice recipe')
        root.geometry('980x640')
        left = ttk.Frame(root, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(root, padding=8)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.avail_title = ttk.Label(left, text='On the rack')
        self.avail_title.grid(row=0, column=0, sticky='w')
        ttk.Label(left, text='Recipe (in this order)').grid(row=0, column=2, sticky='w')
        self.avail = tk.Listbox(left, height=16, exportselection=False)
        self.avail.grid(row=1, column=0, rowspan=6, sticky='ns')
        self.avail.bind('<Double-Button-1>', lambda e: self.add())
        mid = ttk.Frame(left, padding=(6, 0))
        mid.grid(row=1, column=1, rowspan=6)
        ttk.Button(mid, text='Add →', command=self.add).pack(fill=tk.X, pady=2)
        ttk.Button(mid, text='← Remove', command=self.remove).pack(fill=tk.X, pady=2)
        ttk.Button(mid, text='Up', command=lambda: self.move(-1)).pack(fill=tk.X, pady=(12, 2))
        ttk.Button(mid, text='Down', command=lambda: self.move(1)).pack(fill=tk.X, pady=2)
        ttk.Button(mid, text='Clear', command=self.clear).pack(fill=tk.X, pady=(12, 2))
        self.scan_btn = ttk.Button(mid, text='Rescan rack', command=self.rescan)
        self.scan_btn.pack(fill=tk.X, pady=(24, 2))
        self.recipe = tk.Listbox(left, height=16, width=24, exportselection=False)
        self.recipe.grid(row=1, column=2, rowspan=6, sticky='ns')
        self.recipe.bind('<Double-Button-1>', lambda e: self.remove())

        opts = ttk.LabelFrame(left, text='Each spice', padding=6)
        opts.grid(row=7, column=0, columnspan=3, sticky='we', pady=(10, 0))
        self.mode = tk.StringVar(value=next(iter(MODES)))
        ttk.Combobox(opts, textvariable=self.mode, values=list(MODES), state='readonly', width=34).pack(anchor='w')
        self.go_home = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text='go home after each spice / after a failure', variable=self.go_home).pack(anchor='w')
        self.dry = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text='dry run (no motion: to check the recipe)', variable=self.dry).pack(anchor='w')
        self.debug = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="debug: answer the robot's questions myself", variable=self.debug,
                        command=self.show_ask).pack(anchor='w')

        run = ttk.Frame(left, padding=(0, 10))
        run.grid(row=8, column=0, columnspan=3, sticky='we')
        self.start_btn = ttk.Button(run, text='▶ Start', command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=2)
        self.pause_btn = ttk.Button(run, text='Pause', command=lambda: self.send(' '), state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=2)
        self.cont_btn = ttk.Button(run, text='Continue', command=lambda: self.send('\n'), state=tk.DISABLED)
        self.cont_btn.pack(side=tk.LEFT, padx=2)
        self.stop_btn = ttk.Button(run, text='■ Stop', command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        self.ask = ttk.LabelFrame(right, text='The robot asks (debug)', padding=6)
        self.question = ttk.Label(self.ask, text='(nothing)', wraplength=560)
        self.question.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.no_btn = ttk.Button(self.ask, text='No', command=lambda: self.answer('n'), state=tk.DISABLED)
        self.no_btn.pack(side=tk.RIGHT, padx=2)
        self.yes_btn = ttk.Button(self.ask, text='Yes', command=lambda: self.answer('y'), state=tk.DISABLED)
        self.yes_btn.pack(side=tk.RIGHT, padx=2)
        self.state_lbl = ttk.Label(right, text='Pick spices, then Start. Keep a hand on the e-stop.',
                                   font=('TkDefaultFont', 11, 'bold'))
        self.state_lbl.pack(anchor='w', pady=(0, 4))
        self.log = tk.Text(right, wrap='word', height=30)
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.tag_config('fail', foreground='red')
        self.log.tag_config('ok', foreground='dark green')
        self.log.tag_config('head', foreground='blue')
        root.protocol('WM_DELETE_WINDOW', self.quit)
        root.after(100, self.poll)
        if scan:
            root.after(300, self.rescan)
        else:
            self.fill_avail(self.spices, 'Spices (all known: no scan)')

    def show_ask(self):
        if self.debug.get():
            self.ask.pack(fill=tk.X, before=self.state_lbl, pady=(0, 8))
        else:
            self.ask.pack_forget()

    # ------------------------------------------------------------------ recipe list
    def add(self):
        for i in self.avail.curselection():
            name = self.avail.get(i)
            if name not in self.spices:            # "? (label not read)", "(scanning ...)"
                continue
            self.recipe.insert(tk.END, name)
            self.status.append('')

    def fill_avail(self, names, title, unread=0):
        self.avail.delete(0, tk.END)
        for n in names:
            self.avail.insert(tk.END, n)
        for _ in range(unread):
            self.avail.insert(tk.END, '? (label not read)')
            self.avail.itemconfig(tk.END, foreground='grey')
        self.avail_title.config(text=title)

    # ------------------------------------------------------------------ scan
    def rescan(self):
        if self.running:
            return
        self.fill_avail([], 'On the rack: looking ...')
        self.avail.insert(tk.END, '(scanning ...)')
        self.scan_text = ''
        args = [sys.executable, os.path.join(ROOT, 'run_task.py'), '--scan']
        if self.dry.get():
            args.append('--dry-run')
        self.write('\n===== scan: which spices are on the rack =====\n', 'head')
        self.state_lbl.config(text='Looking at the rack (the arm goes to the survey pose) ...')
        self.busy(True)
        self.launch(args, 'scan')

    def scan_ended(self, code):
        found = None
        for line in self.scan_text.splitlines():
            if line.startswith('SCAN_RESULT '):
                try:
                    found = json.loads(line[len('SCAN_RESULT '):])['lids']
                except (ValueError, KeyError):
                    pass
        self.busy(False)
        if code != 0 or found is None:
            self.fill_avail(self.spices, 'Spices (scan failed: all known)')
            self.finish('The scan failed (see the log): all known spices listed. Fix it and Rescan, '
                        'or pick anyway.', 'fail')
            return
        names = [f['label'] for f in found if f['label']]
        unread = sum(1 for f in found if not f['label'])
        self.fill_avail(names, f'On the rack ({len(found)} jar{"s" if len(found) != 1 else ""})', unread)
        msg = (f'Found: {", ".join(names) or "no readable labels"}' +
               (f'; {unread} label(s) not read (turn them to face up, Rescan)' if unread else '') +
               '. Pick spices, then Start.')
        self.finish(msg, 'ok' if names else 'fail')

    def remove(self):
        if self.running:
            return
        for i in reversed(self.recipe.curselection()):
            self.recipe.delete(i)
            del self.status[i]

    def move(self, d):
        if self.running:
            return
        sel = self.recipe.curselection()
        if not sel:
            return
        i, j = sel[0], sel[0] + d
        if not 0 <= j < self.recipe.size():
            return
        names = list(self.recipe.get(0, tk.END))
        names[i], names[j] = names[j], names[i]
        self.status[i], self.status[j] = self.status[j], self.status[i]
        self.redraw(names)
        self.recipe.selection_set(j)

    def clear(self):
        if not self.running:
            self.recipe.delete(0, tk.END)
            self.status = []

    def names(self):
        return [n.split('  ')[0] for n in self.recipe.get(0, tk.END)]

    def redraw(self, names=None):
        names = names or self.names()
        names = [n.split('  ')[0] for n in names]
        self.recipe.delete(0, tk.END)
        for n, st in zip(names, self.status):
            self.recipe.insert(tk.END, n + (f'  [{st}]' if st else ''))
            colour = {'done': 'dark green', 'FAILED': 'red', 'running': 'blue'}.get(st)
            if colour:
                self.recipe.itemconfig(tk.END, foreground=colour)

    # ------------------------------------------------------------------ running
    def start(self):
        if self.running:
            return
        if not self.names():
            self.state_lbl.config(text='The recipe is empty: add spices first.')
            return
        self.status = ['' if s != 'done' else s for s in self.status]
        if all(s == 'done' for s in self.status):        # all done: a new round
            self.status = [''] * len(self.status)
        self.busy(True)
        self.next_spice()

    def next_spice(self):
        todo = [i for i, s in enumerate(self.status) if s != 'done']
        if not todo:
            self.finish('Recipe done: every spice poured and back in the rack.', 'ok')
            return
        self.current = i = todo[0]
        spice = self.names()[i]
        self.status[i] = 'running'
        self.redraw()
        args = [sys.executable, os.path.join(ROOT, 'run_task.py'), '--spice', spice] + MODES[self.mode.get()]
        if self.go_home.get():
            args.append('--home')
        if self.dry.get():
            args.append('--dry-run')
        self.write(f'\n===== {spice} ({i + 1}/{len(self.status)}): {" ".join(args[2:])} =====\n', 'head')
        self.state_lbl.config(text=f'Running {spice} ({i + 1} of {len(self.status)})')
        self.launch(args, 'spice')

    def launch(self, args, job):
        self.job = job
        self.master, slave = pty.openpty()
        self.proc = subprocess.Popen(args, stdin=slave, stdout=slave, stderr=slave, cwd=ROOT,
                                     start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED='1'))
        os.close(slave)
        threading.Thread(target=self.reader, args=(self.master, self.proc), daemon=True).start()

    def busy(self, on):
        self.running = on
        for b in (self.pause_btn, self.cont_btn, self.stop_btn):
            b.config(state=tk.NORMAL if on else tk.DISABLED)
        for b in (self.start_btn, self.scan_btn):
            b.config(state=tk.DISABLED if on else tk.NORMAL)

    def reader(self, fd, proc):
        while True:
            try:
                ready, _, _ = select.select([fd], [], [], 0.2)
                if ready:
                    data = os.read(fd, 4096)
                    if not data:
                        break
                    self.out.put(('text', data.decode(errors='replace')))
                elif proc.poll() is not None:
                    break
            except OSError:
                break
        proc.wait()
        self.out.put(('exit', proc.returncode))

    def poll(self):
        try:
            while True:
                kind, val = self.out.get_nowait()
                if kind == 'text':
                    self.handle_text(val)
                else:
                    self.run_ended(val)
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def handle_text(self, text):
        text = ANSI.sub('', text).replace('\r\n', '\n').replace('\r', '')
        if self.job == 'scan':
            self.scan_text += text
        tag = 'fail' if 'FAILED' in text or 'WARNING' in text else None
        self.write(text, tag)
        tail = self.log.get('end-2l', 'end-1c').strip()
        asking = tail.endswith('[y/N]')
        if asking and not self.prompt:
            self.prompt = True
            if not self.debug.get():                 # automatic: the safe answer
                self.write('  -> answered No automatically (the safe choice)\n', 'head')
                self.answer('n')
                return
            self.question.config(text=tail.rsplit('\n', 1)[-1])
            self.yes_btn.config(state=tk.NORMAL)
            self.no_btn.config(state=tk.NORMAL)
            self.root.bell()
            self.root.deiconify()

    def answer(self, key):
        if not self.prompt:
            return
        self.prompt = False
        self.question.config(text='(nothing)')
        self.yes_btn.config(state=tk.DISABLED)
        self.no_btn.config(state=tk.DISABLED)
        self.send(key + '\n')

    def send(self, keys):
        if self.master is not None and self.proc is not None and self.proc.poll() is None:
            try:
                os.write(self.master, keys.encode())
            except OSError:
                pass

    def stop(self):
        """Like q in the terminal (the run stops, asks about going home); twice: Ctrl-C."""
        if self.proc is None or self.proc.poll() is not None:
            return
        if getattr(self, '_stopping', False):
            os.killpg(self.proc.pid, signal.SIGINT)
        else:
            self._stopping = True
            self.send(' ')                      # pause first (cancels the move in progress) ...
            self.root.after(300, lambda: self.send('q'))   # ... then stop the run
        self.state_lbl.config(text='Stopping ... (Stop again = Ctrl-C)')

    def run_ended(self, code):
        self._stopping = False
        if self.master is not None:
            try:
                os.close(self.master)
            except OSError:
                pass
            self.master = None
        self.answer_clear()
        if self.job == 'scan':
            self.scan_ended(code)
            return
        i = self.current
        if code == 0:
            self.status[i] = 'done'
            self.redraw()
            self.write(f'----- {self.names()[i]}: done\n', 'ok')
            self.next_spice()
        else:
            self.status[i] = 'FAILED'
            self.redraw()
            self.finish(f'{self.names()[i]} FAILED (see the log). Fix it, then Start carries on from it '
                        '(or remove it from the recipe).', 'fail')

    def answer_clear(self):
        self.prompt = False
        self.question.config(text='(nothing)')
        self.yes_btn.config(state=tk.DISABLED)
        self.no_btn.config(state=tk.DISABLED)

    def finish(self, msg, tag):
        self.busy(False)
        self.proc = None
        self.state_lbl.config(text=msg)
        self.write(msg + '\n', tag)

    def write(self, text, tag=None):
        self.log.insert(tk.END, text, tag)
        self.log.see(tk.END)

    def quit(self):
        if self.proc is not None and self.proc.poll() is None:
            self.state_lbl.config(text='A spice is still running: Stop it first.')
            return
        self.root.destroy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=os.path.join(ROOT, 'config', 'real.yaml'))
    p.add_argument('--no-scan', action='store_true', help='do not look at the rack first: list every known spice')
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    root = tk.Tk()
    RecipeApp(root, cfg, scan=not args.no_scan)
    root.mainloop()


if __name__ == '__main__':
    main()
