"""Motor TTY in a separate process, independent of the vision interpreter.
Fresh command lease: zero after 300ms, disarm after 600ms. UNO watchdog: 350ms.
Construct before camera/display initialization; only the child uses the TTY.
"""

import multiprocessing
import time

from robot.core.road_policy import MAX_PREDICTION_COMMAND_S


def _motor_error(state, message):
    data = str(message).encode("utf-8")[:255]
    state["error"].value = data


def _motor_exchange(transport, state, op, left=0, right=0):
    ack = transport.send(op, left, right)
    now = time.monotonic()
    with state["lock"]:
        v = state["values"]
        # Count only short intervals bracketed by successful forward-output
        # ACKs. Zero, arm/stop, faults and >=300ms gaps break the chain. This
        # deliberately undercounts uncertain/start/stop intervals.
        forward = op == 2 and ack[1] == 1 and ack[2] + ack[3] > 0 and ack[4] == 0
        if forward:
            interval = now - v[14]
            if v[14] > 0 and 0 < interval < 0.30:
                v[13] += interval
            v[14] = now
        else:
            v[14] = 0.0
        v[4] = now
        v[7:12] = ack[:5]
        state["lamp"].value = ack[5] if len(ack) == 6 else 0
        if op == 2:
            if v[6]:
                v[5] = max(v[5], now - v[6])
            v[6] = now
    return ack


def _motor_fail(transport, state, exc):
    with state["lock"]:
        v = state["values"]
        v[0] = 0
        v[1] = v[2] = 0
        v[14] = 0.0
        _motor_error(state, exc)
    try:
        _motor_exchange(transport, state, 0)
    except Exception:
        pass


def _motor_worker(transport, state, connection, closed, period, fresh_seconds, stop_seconds):
    next_drive = 0.0
    try:
        while not closed.is_set():
            if connection.poll(0.005):
                op, deadline = connection.recv()
                try:
                    if time.monotonic() > deadline:
                        raise RuntimeError("Expired motor request")
                    ack = _motor_exchange(transport, state, op)
                    if op in (0, 1):
                        with state["lock"]:
                            v = state["values"]
                            v[0] = int(op == 1)
                            v[1] = v[2] = 0
                            v[3] = time.monotonic()
                            v[6] = 0
                            _motor_error(state, "")
                        next_drive = 0.0
                    connection.send((ack, None))
                except Exception as exc:
                    _motor_fail(transport, state, exc)
                    connection.send((None, str(exc)))
            with state["lock"]:
                v = state["values"]
                armed = bool(v[0])
                target = (int(v[1]), int(v[2]))
                published = v[3]
                lease_until = v[12]
            now = time.monotonic()
            if not armed or now < next_drive:
                continue
            age = now - published
            if age >= stop_seconds:
                _motor_fail(transport, state, "VISION COMMAND EXPIRED")
                continue
            try:
                output = (
                    target
                    if age < fresh_seconds and (not lease_until or now < lease_until)
                    else (0, 0)
                )
                _motor_exchange(transport, state, 2, *output)
                next_drive = time.monotonic() + period
                if lease_until and output != (0, 0):
                    next_drive = min(next_drive, lease_until)
            except Exception as exc:
                _motor_fail(transport, state, exc)
    finally:
        try:
            _motor_exchange(transport, state, 0)
        except Exception:
            pass
        transport.port.close()
        connection.close()


class MotorChannel:
    def __init__(self, transport, period=0.04, fresh_seconds=0.30, stop_seconds=0.60):
        self.transport = transport
        self.fresh_seconds = float(fresh_seconds)
        # Linux device: fork before any camera/display resources exist. The child
        # uses POSIX TTY calls only, never inherited Maix camera or display APIs.
        ctx = multiprocessing.get_context("fork")
        self.state = dict(
            values=ctx.RawArray("d", 15),
            error=ctx.RawArray("c", 256),
            lock=ctx.Lock(),
            lamp=ctx.RawValue("i", 0),
        )
        self.closed = ctx.Event()
        self.connection, child = ctx.Pipe()
        self.worker = ctx.Process(
            target=_motor_worker,
            args=(transport, self.state, child, self.closed, period, fresh_seconds, stop_seconds),
            daemon=True,
        )
        self.worker.start()
        child.close()
        self.unusable = False

    @property
    def port(self):
        return self

    def snapshot(self):
        with self.state["lock"]:
            v = list(self.state["values"])
            error = self.state["error"].value.decode("utf-8") or None
            lamp_flags = self.state["lamp"].value
        return dict(
            armed=bool(v[0]),
            confirmed_forward_s=v[13],
            ack=[int(x) for x in v[7:12]],
            error=error,
            brake_lamp_supported=bool(lamp_flags & 1),
            brake_lamp_on=bool(lamp_flags & 2),
            ack_age_ms=round((time.monotonic() - v[4]) * 1000),
            max_drive_gap_ms=round(v[5] * 1000, 1),
            worker_alive=self.worker.is_alive(),
        )

    def send(self, op, left=0, right=0, *, lease_s=None):
        if self.unusable or self.closed.is_set() or not self.worker.is_alive():
            raise RuntimeError("Motor process unavailable")
        if op == 2:
            # Recovery rotations retain their short limit; forward prediction
            # may bridge one camera interval within the independent watchdog.
            maximum = min(self.fresh_seconds, MAX_PREDICTION_COMMAND_S)
            if left * right < 0:
                maximum = min(maximum, 0.15)
            if lease_s is not None and not 0 < float(lease_s) <= maximum:
                raise ValueError("Invalid bounded motion lease")
            with self.state["lock"]:
                v = self.state["values"]
                error = self.state["error"].value.decode("utf-8")
                if error or not v[0]:
                    raise RuntimeError(error or "Motor channel disarmed")
                v[1] = int(left)
                v[2] = int(right)
                v[3] = time.monotonic()
                v[12] = 0.0 if lease_s is None else v[3] + lease_s
            return None  # A published target; snapshot reports the actual UNO ACK.
        self.connection.send((op, time.monotonic() + 0.75))
        if not self.connection.poll(0.8):
            self.unusable = True
            self.closed.set()
            raise RuntimeError("Motor process request timed out")
        ack, error = self.connection.recv()
        if error:
            raise RuntimeError(error)
        return ack

    def stop(self):
        try:
            return self.send(0)
        except Exception as exc:
            return "STOP ACK unavailable: " + str(exc)

    def close(self):
        self.closed.set()
        self.worker.join(timeout=0.8)
        if self.worker.is_alive():
            raise RuntimeError("Motor process did not stop")
        self.connection.close()
        self.transport.port.close()
