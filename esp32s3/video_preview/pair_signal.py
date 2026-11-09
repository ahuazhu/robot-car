"""Pair ESP32 to Maix using USB, writing a private local config (never print key)."""
import argparse
import json
import os
from pathlib import Path
import secrets
import time
from urllib.parse import urlsplit
import serial

if __package__:
    from .device_protocol import supported_firmware
else:
    from device_protocol import supported_firmware


def pair(port, peer_url, destination):
    url=urlsplit(peer_url)
    if url.scheme!='http' or not url.hostname or url.username or url.password or url.path not in ('','/') or url.query or url.fragment:
        raise ValueError('Expected ESP32 HTTP base URL')
    config=dict(peer_url=peer_url.rstrip('/'),key=secrets.token_hex(32))
    with serial.Serial(port,115200,timeout=.25,write_timeout=2,exclusive=True) as link:
        time.sleep(3)
        link.reset_input_buffer();link.write(b'Q')
        deadline=time.monotonic()+5;verified=False
        while time.monotonic()<deadline:
            line=link.readline(1024)
            if line.startswith(b'WIFI '):
                state=json.loads(line[5:]);verified=isinstance(state,dict) and supported_firmware(state.get('firmware'));break
        if not verified:raise RuntimeError('ESP32 start-signal firmware not detected')
        link.write(b'P'+json.dumps({'key':config['key']}).encode()+b'\n')
        deadline=time.monotonic()+5;paired=False
        while time.monotonic()<deadline:
            line=link.readline(1024)
            if line.startswith(b'PAIR '):paired=line.strip()==b'PAIR OK';break
        if not paired:raise RuntimeError('ESP32 pairing was not acknowledged')
    path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as output:json.dump(config,output)
    tmp.chmod(0o600);os.replace(tmp,path)
    print('ESP32 paired. Private config saved; key is not printed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial',required=True)
    parser.add_argument('--peer-url',required=True)
    parser.add_argument('--output',type=Path,default=Path('.maixpy/start_signal.local.json'))
    args=parser.parse_args();pair(args.serial,args.peer_url,args.output)
