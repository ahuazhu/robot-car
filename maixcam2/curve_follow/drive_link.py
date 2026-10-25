import time
import os
import select
import json
import termios
from controller import MAX_MOTOR_PWM
def crc8(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ (7 if crc & 128 else 0)) & 255
    return crc

class NonblockingUART:
    """Linux TTY backend; select/os.read release the interpreter during waits."""
    def __init__(self, path):
        self.fd=os.open(path,os.O_RDWR|os.O_NOCTTY|os.O_NONBLOCK)
        try:
            attrs=termios.tcgetattr(self.fd)
            attrs[0]=0;attrs[1]=0;attrs[2]=termios.CLOCAL|termios.CREAD|termios.CS8
            attrs[3]=0;attrs[4]=termios.B9600;attrs[5]=termios.B9600
            attrs[6][termios.VMIN]=0;attrs[6][termios.VTIME]=0
            termios.tcsetattr(self.fd,termios.TCSANOW,attrs)
            termios.tcflush(self.fd,termios.TCIOFLUSH)
        except Exception:
            os.close(self.fd);self.fd=None;raise
    def read(self, timeout=0):
        if self.fd is None:raise RuntimeError('UART closed')
        if timeout and not select.select([self.fd],[],[],timeout)[0]:return b''
        try:return os.read(self.fd,256)
        except BlockingIOError:return b''
    def write_str(self, text):
        data=text.encode('ascii');deadline=time.monotonic()+.1
        while data:
            if time.monotonic()>=deadline:raise RuntimeError('UART write timeout')
            try:
                count=os.write(self.fd,data)
                if not count:raise RuntimeError('UART write failed')
                data=data[count:]
            except BlockingIOError:select.select([],[self.fd],[],.01)
    def close(self):
        if self.fd is not None:
            os.close(self.fd);self.fd=None

class DriveLink:
    def __init__(self):
        from maix import uart, pinmap, err
        err.check_raise(pinmap.set_pin_function('A21', 'UART4_TX'), 'UART4 TX')
        err.check_raise(pinmap.set_pin_function('A22', 'UART4_RX'), 'UART4 RX')
        # Construct once to release Maix's default UART4 communication service.
        setup_port = uart.UART('/dev/ttyS4', 9600)
        setup_port.close()
        self.port = NonblockingUART('/dev/ttyS4')
        self.seq = 0

    def send(self, op, left=0, right=0):
        if type(op) is not int or op not in (0,1,2,3):raise ValueError('Invalid motor operation')
        if any(type(v) is not int or abs(v)>MAX_MOTOR_PWM for v in (left,right)):
            raise ValueError('Motor command outside configured PWM range')
        if op!=2 and (left or right):raise ValueError('Non-drive operation requires zero PWM')
        self.seq = (self.seq + 1) & 65535
        payload = ('%d,%d,%d,%d' % (self.seq, op, left, right)).encode('ascii')
        self.port.read(timeout=0)
        tx='@' + payload.decode('ascii') + '*%02X\n' % crc8(payload)
        self.last_error_diagnostics=None
        self.port.write_str(tx)
        deadline = time.monotonic() + 0.30
        buf = b'';received=bytearray()
        while time.monotonic() < deadline:
            data = self.port.read(timeout=0)
            if data:
                received.extend(data)
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
                        if len(values) in (5,6) and values[0] == self.seq:
                            if len(values)==6 and (values[5] not in (4,5,7) or (values[2] or values[3]) and values[5]&2):
                                raise RuntimeError('Invalid brake lamp status')
                            if values[1] not in (0,1) or any(abs(v)>MAX_MOTOR_PWM for v in values[2:4]):
                                raise RuntimeError('Invalid UNO status')
                            if op == 0 and values[1:4] != [0,0,0]:
                                raise RuntimeError('UNO did not confirm stopped')
                            if op in (1,2) and (values[1] != 1 or values[4] != 0 or values[2:4] != [left,right]):
                                raise RuntimeError('UNO rejected motion: %s' % values)
                            return values
                    except (ValueError, IndexError):
                        continue
            time.sleep(0.002)
        self.last_error_diagnostics=dict(sequence=self.seq,operation=op,tx=tx.strip(),
                                         rx_bytes=len(received),rx_hex=bytes(received).hex())
        print('UART_TIMEOUT',json.dumps(self.last_error_diagnostics),flush=True)
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
