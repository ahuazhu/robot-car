"""Run ON MaixCAM2 via MaixVision; this is not a desktop Python program.

Side 6-pin UART4, A21=TX, A22=RX. UNO must run uno_uart_test.ino.
The 9600 baud link is deliberately slow for initial SoftwareSerial testing.
USB Serial Monitor on the UNO uses a separate baud rate of 115200.
"""
import time
from maix import app, uart, pinmap, err

BAUD = 9600


def main():
    err.check_raise(pinmap.set_pin_function("A21", "UART4_TX"), "Map A21 TX")
    err.check_raise(pinmap.set_pin_function("A22", "UART4_RX"), "Map A22 RX")
    link = uart.UART("/dev/ttyS4", BAUD)
    print("MAIX READY | UART4 TX=A21 RX=A22 baud=9600", flush=True)
    sequence = 0
    passed = 0
    failed = 0
    buffer = b""
    dropping = False
    try:
        while not app.need_exit():
            sequence += 1
            # Discard stale bytes before starting a new round trip.
            link.read(timeout=0)
            buffer = b""
            dropping = False
            payload = "PING %d\n" % sequence
            expected = ("PONG %d" % sequence).encode("ascii")
            print("TX:", payload.strip(), flush=True)
            link.write_str(payload)
            started = time.monotonic()
            matched = False
            while not app.need_exit() and time.monotonic() - started < 0.8:
                data = link.read(timeout=0)
                if data:
                    for value in data:
                        if value == 10:
                            if not dropping:
                                message = buffer.rstrip(b"\r")
                                print("RX:", repr(message), flush=True)
                                if message == expected:
                                    matched = True
                            buffer = b""
                            dropping = False
                        elif not dropping:
                            if len(buffer) < 64:
                                buffer += bytes([value])
                            else:
                                buffer = b""
                                dropping = True
                if matched:
                    break
                time.sleep(0.005)
            if app.need_exit():
                break
            if matched:
                passed += 1
                print("PASS %d | total_ok=%d total_timeout=%d" %
                      (sequence, passed, failed), flush=True)
            else:
                failed += 1
                print("TIMEOUT %d | total_ok=%d total_timeout=%d" %
                      (sequence, passed, failed), flush=True)
            # Send only one request at a time; SoftwareSerial is half duplex.
            while not app.need_exit() and time.monotonic() - started < 1.0:
                time.sleep(0.01)
    finally:
        link.close()


if __name__ == "__main__":
    main()
