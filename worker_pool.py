# MeetingGenie - Persistent Worker Pool
#
# Manages a single long-lived multiprocessing worker process per backend
# (whisper-CT2, whisper-MLX, pyannote). The worker imports heavy libraries
# (torch, faster_whisper, ...) once, loads the model once, and then
# processes jobs in a loop — keeping the model resident in memory.
#
# Protocol (parent -> worker, on `in_q`):
#   {"cmd": "init", "args": {...}}    # one-time worker init / model load
#   {"cmd": "job",  "args": {...}}    # run one transcription job
#   {"cmd": "ping"}                    # health check
#   {"cmd": "cancel"}                  # set a cancel flag for current job
#   {"cmd": "exit"}                    # break the loop and exit cleanly
#
# Protocol (worker -> parent, on the SHARED out_q passed at start):
#   Same shape as the existing spawn-per-job workers
#   ({"type": "log"|"progress"|"segment"|"device"|"result"}).
#   The worker additionally emits {"type": "ready"} after INIT succeeds
#   and {"type": "pong"} in response to a ping.
#
# A single shared output queue is used per worker. Spawn-context queues
# can't be pickled inside command messages — they only cross the
# process boundary via inheritance. The orchestrator submits one job at
# a time and drains the shared queue to a {"type": "result"} terminator
# before submitting the next, so message ordering is unambiguous.

import multiprocessing as mp
import queue as pyqueue
import threading
import time
import traceback
from typing import Optional


class WorkerPool:
    """One long-lived subprocess that handles many sequential jobs.

    Not a "pool" in the multi-worker sense — just a single persistent
    worker process. The name reflects intent (a managed pool we can
    grow later) rather than current cardinality.
    """

    # ---- lifecycle --------------------------------------------------

    def __init__(self, target, init_args: dict, name: str = "worker"):
        """
        target      — callable used as the subprocess target. Must accept
                       (in_q) and run the persistent worker loop. The
                       worker reads command messages from in_q and emits
                       output messages on the per-job queue carried in
                       each "job" / "init" / "ping" message.
        init_args   — sent as the payload of the initial INIT command.
        name        — for log messages.
        """
        self.target = target
        self.init_args = init_args
        self.name = name

        self._ctx = mp.get_context("spawn")
        self._in_q = None
        self._out_q = None  # shared output queue (inherited by worker)
        self._proc = None
        self._lock = threading.Lock()
        self._alive = False
        self._started_at = 0.0

    def start(self, timeout: float = 60.0) -> bool:
        """Spawn the worker and wait for it to confirm the model is loaded."""
        with self._lock:
            if self._alive:
                return True
            self._in_q = self._ctx.Queue()
            self._out_q = self._ctx.Queue()
            self._proc = self._ctx.Process(
                target=self.target,
                args=(self._in_q, self._out_q),
                daemon=True,
            )
            self._proc.start()
            self._started_at = time.time()

            # Send INIT — output (ready/error) arrives on the shared out_q.
            try:
                self._in_q.put({"cmd": "init", "args": self.init_args})
            except Exception as e:
                print(f"[worker_pool:{self.name}] failed to send init: {e}")
                self._alive = False
                return False

            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    msg = self._out_q.get(timeout=0.25)
                except pyqueue.Empty:
                    if not self._proc.is_alive():
                        print(f"[worker_pool:{self.name}] worker died during init (code {self._proc.exitcode})")
                        self._alive = False
                        return False
                    continue
                mtype = msg.get("type")
                if mtype == "ready":
                    self._alive = True
                    elapsed = time.time() - self._started_at
                    print(f"[worker_pool:{self.name}] ready in {elapsed:.1f}s")
                    return True
                if mtype == "result" and not msg.get("ok", False):
                    print(f"[worker_pool:{self.name}] init failed: {msg.get('error')}")
                    self._alive = False
                    return False
                # ignore log/progress noise during init
            print(f"[worker_pool:{self.name}] init timed out after {timeout}s")
            self._alive = False
            return False

    def is_alive(self) -> bool:
        if not self._alive or self._proc is None:
            return False
        if not self._proc.is_alive():
            self._alive = False
            return False
        return True

    def ping(self, timeout: float = 5.0) -> bool:
        if not self.is_alive():
            return False
        try:
            self._in_q.put({"cmd": "ping"})
        except Exception:
            return False
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = self._out_q.get(timeout=0.25)
            except pyqueue.Empty:
                if not self.is_alive():
                    return False
                continue
            if msg.get("type") == "pong":
                return True
        return False

    # ---- job submission ---------------------------------------------

    def run_job(self, args: dict):
        """Submit a job to the worker. Returns the shared output queue
        the orchestrator can drain with the same protocol as today's
        spawn-per-job workers.

        Only one job may be in flight at a time. The caller is
        responsible for draining the queue until a {"type": "result"}
        message arrives before submitting the next job.
        """
        if not self.is_alive():
            raise RuntimeError(f"WorkerPool '{self.name}' is not alive")
        self._in_q.put({"cmd": "job", "args": args})
        return self._out_q

    def cancel_current(self):
        """Best-effort: ask the worker to abort the current job.
        Most transcription loops can't cleanly abort mid-segment — in
        that case the orchestrator should terminate the worker and
        rely on `is_alive()` to fall back to spawn-per-job on the next
        invocation.
        """
        if not self.is_alive():
            return
        try:
            self._in_q.put({"cmd": "cancel"})
        except Exception:
            pass

    # ---- shutdown ---------------------------------------------------

    def shutdown(self, timeout: float = 3.0):
        with self._lock:
            if self._proc is None:
                return
            if self._alive:
                try:
                    self._in_q.put({"cmd": "exit"})
                except Exception:
                    pass
            self._alive = False
            try:
                self._proc.join(timeout=timeout)
            except Exception:
                pass
            if self._proc.is_alive():
                try:
                    self._proc.terminate()
                except Exception:
                    pass
                try:
                    self._proc.join(timeout=1.0)
                except Exception:
                    pass
            self._proc = None
            self._in_q = None


# ---- worker-side loop helper ---------------------------------------

def run_worker_loop(in_q, out_q, init_fn, job_fn):
    """Generic loop the worker subprocess runs.

    in_q                            — receives command dicts from parent.
    out_q                           — shared queue, all responses go here.
    init_fn(init_args) -> state     — called once on INIT. Raises on failure.
    job_fn(state, args, out_q, flag)— per job; emits messages on out_q
                                       (terminating with {"type": "result"}).

    The loop also responds to ping/cancel/exit commands.
    """
    state = None
    initialized = False
    cancel_flag = {"set": False}

    while True:
        try:
            msg = in_q.get()
        except (EOFError, KeyboardInterrupt):
            return
        if not isinstance(msg, dict):
            continue
        cmd = msg.get("cmd")

        if cmd == "init":
            try:
                state = init_fn(msg.get("args", {}))
                initialized = True
                try:
                    out_q.put({"type": "ready"})
                except Exception:
                    pass
            except Exception as e:
                try:
                    out_q.put({
                        "type": "result",
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                        "trace": traceback.format_exc(),
                    })
                except Exception:
                    pass
                # Without an initialized state we can't do anything useful;
                # exit so the parent notices the pool is dead.
                return

        elif cmd == "ping":
            try:
                out_q.put({"type": "pong"})
            except Exception:
                pass

        elif cmd == "cancel":
            cancel_flag["set"] = True

        elif cmd == "job":
            if not initialized or state is None:
                try:
                    out_q.put({
                        "type": "result",
                        "ok": False,
                        "error": "worker not initialized",
                        "trace": "",
                    })
                except Exception:
                    pass
                continue
            cancel_flag["set"] = False
            try:
                job_fn(state, msg.get("args", {}), out_q, cancel_flag)
            except Exception as e:
                # Never let a job error crash the loop.
                try:
                    out_q.put({
                        "type": "result",
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                        "trace": traceback.format_exc(),
                    })
                except Exception:
                    pass

        elif cmd == "exit":
            return
