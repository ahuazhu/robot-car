import time
def crc8(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ (7 if crc & 128 else 0)) & 255
    return crc

class DriveLink:
    def __init__(self):
        from maix import uart, pinmap, err
        err.check_raise(pinmap.set_pin_function('A21', 'UART4_TX'), 'UART4 TX')
        err.check_raise(pinmap.set_pin_function('A22', 'UART4_RX'), 'UART4 RX')
        self.port = uart.UART('/dev/ttyS4', 9600)
        self.seq = 0

    def send(self, op, left=0, right=0):
        self.seq = (self.seq + 1) & 65535
        payload = ('%d,%d,%d,%d' % (self.seq, op, left, right)).encode('ascii')
        self.port.read(timeout=0)
        self.port.write_str('@' + payload.decode('ascii') + '*%02X\n' % crc8(payload))
        deadline = time.monotonic() + 0.16
        buf = b''
        while time.monotonic() < deadline:
            data = self.port.read(timeout=0)
            if data:
                buf += data
                if len(buf) > 256:
                    raise RuntimeError('UART receive overflow')
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    line = line.rstrip(b'\r')
                    try:
                        body, checksum = line[1:].split(b'*')
                        if not line.startswith(b'!') or len(checksum) != 2 or crc8(body) != int(checksum, 16):
                            continue
                        values = [int(v) for v in body.split(b',')]
                        if len(values) == 5 and values[0] == self.seq:
                            if values[1] not in (0,1) or any(abs(v)>120 for v in values[2:4]):
                                raise RuntimeError('Invalid UNO status')
                            if op == 0 and values[1:4] != [0,0,0]:
                                raise RuntimeError('UNO did not confirm stopped')
                            if op in (1,2) and (values[1] != 1 or values[4] != 0 or values[2:4] != [left,right]):
                                raise RuntimeError('UNO rejected motion: %s' % values)
                            return values
                    except (ValueError, IndexError):
                        continue
            time.sleep(0.002)
        raise RuntimeError('UNO acknowledgement timeout')

    def stop(self):
        try:
            return self.send(0)
        except Exception as exc:
            print('STOP ACK unavailable; UNO watchdog will stop: %s' % exc, flush=True)
            return None

    def close(self):
        self.stop()
        self.port.close()
