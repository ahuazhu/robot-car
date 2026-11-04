"""A deterministic UART peer and clock for protocol tests."""

from robot.control import drive_link


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        self.now += 0.001
        return self.now

    def sleep(self, t):
        self.now += t


class Port:
    def __init__(self, reply):
        self.reply = reply
        self.tx = None

    def read(self, timeout=0):
        if self.tx is None:
            return b""
        reply, self.reply = self.reply, b""
        return reply

    def write_str(self, text):
        self.tx = text


def link(reply):
    obj = drive_link.DriveLink.__new__(drive_link.DriveLink)
    obj.seq = 0
    obj.port = Port(reply)
    return obj


def ack(sequence=1):
    body = ("%d,0,0,0,0" % sequence).encode()
    return b"!" + body + b"*%02X\n" % drive_link.crc8(body)
