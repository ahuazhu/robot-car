"""Independent preview refusal must never stop capture or authorize Maix."""
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from PIL import Image
from esp32s3.lampnet import verify_console as preview

NODE = shutil.which('node')
if not NODE:
    bundled = Path('/Applications/ChatGPT.app/Contents/Resources/cua_node/bin/node')
    if bundled.is_file():
        NODE = str(bundled)


def model(*, manual=True, armed=False, selftest=True, roi=None):
    return dict(selftest_ok=selftest, diagnostic=dict(native_armed=armed, native_confirmed=False,
                enabled=False, configuration=dict(manual=manual, roi=roi or [61, 59, 64, 53])))


JS_CHECKS = r"""
return (async () => {
 const assert=require('node:assert/strict'),vm=require('node:vm');
 const elements=new Map();let fetches=0;
 function element(id){
  if(!elements.has(id))elements.set(id,{id,disabled:false,hidden:false,dataset:{},children:[],style:{},textContent:'',title:'',
   setAttribute(name,value){this[name]=value;},toggleAttribute(name,value){this[name]=value;},
   addEventListener(){},appendChild(child){this.children.push(child);},setPointerCapture(){},
   getBoundingClientRect(){return {left:0,top:0,width:320,height:240};}});
  return elements.get(id);
 }
 const context=vm.createContext({document:{getElementById:element,createElementNS(){return {dataset:{},setAttribute(){}};}},
  history:{replaceState(){}},location:{hash:'',pathname:'/'},URLSearchParams,setInterval(){},
  fetch(){fetches++;throw Error('unexpected network request');}});
 const script=source.match(/<script>([\s\S]*?)<\/script>/)[1].replace('setInterval(poll,400);draw();poll();','');
 vm.runInContext(script,context);
 const state=(manual=true)=>({model:{selftest_ok:false,diagnostic:{native_armed:false,configuration:{manual,roi:[61,59,64,53]}}},
  frame:1,frame_age_ms:0,error:'',control_ready:true,test_ready:true,test_pending:false});
 function render(data){context.input=data;vm.runInContext('renderStatus(input)',context);}
 render(state());
 assert.equal(element('test').disabled,false,'manual mode must not wait for model self-test');
 assert.ok(!element('status').textContent.includes('等待模型'));
 const formal=state();formal.model.diagnostic.native_armed=true;formal.test_ready=false;formal.control_ready=false;formal.test_blocked_reason='Maix 已 ARM';
 render(formal);
 for(const id of ['test','auto','apply'])assert.equal(element(id).disabled,true,'formal ARM must disable '+id);
 assert.ok(element('status').textContent.includes('正式等灯'));
 await vm.runInContext("el('test').onclick()",context);
 assert.equal(fetches,0,'disabled test must not POST');
 const automatic=state(false);automatic.test_ready=false;render(automatic);
 assert.equal(element('test').disabled,true,'automatic mode must still wait for self-test');
 assert.ok(element('status').textContent.includes('模型自检'));
 const pending=state();pending.test_pending=true;render(pending);
 assert.equal(element('test').disabled,true,'pending request cannot be sent twice');
 assert.ok(element('status').textContent.includes('待处理'));
 vm.runInContext('edit=true',context);render(state());
 assert.equal(element('test').disabled,true,'unsaved draft must not be tested');
 vm.runInContext('edit=false;pendingConfig={manual:true,roi:[80,40,60,60]}',context);render(state());
 assert.equal(element('test').disabled,true,'unacknowledged configuration must not be tested');
 vm.runInContext('pendingConfig=null',context);
 const confirmed=state();confirmed.model.diagnostic.confirmed=true;render(confirmed);
 assert.ok(element('status').textContent.includes('不代表 Maix 发车许可'));
 render({model:{},frame:1,frame_age_ms:0,control_ready:false,test_ready:false});
 assert.equal(element('test').disabled,true,'unknown state must fail closed');
 return 'Front-end behaviour checks passed';
})();
"""


class CameraPort:
    def __init__(self, state=None):
        self.state = state or model()
        self.writes, self.buffer = [], b''
        self.closed = False
        self.condition = threading.Condition()
        self.images = 0
        output = io.BytesIO()
        Image.new('RGB', (320, 240), (35, 50, 80)).save(output, 'JPEG')
        self.jpeg = output.getvalue()

    def open(self):
        pass

    def write(self, data):
        with self.condition:
            self.writes.append(data)
            if data == b'M\n':
                self.buffer += b'LAMP_MODEL '+json.dumps(self.state).encode()+b'\n'
            elif data == b'J\n':
                self.buffer += b'JPEG %d\n' % len(self.jpeg)+self.jpeg+b'\n'
                self.images += 1
            self.condition.notify_all()
        return len(data)

    def read(self, size):
        with self.condition:
            result, self.buffer = self.buffer[:size], self.buffer[size:]
            return result

    def flush(self):
        pass

    def close(self):
        self.closed = True


class IndependentPreviewTests(unittest.TestCase):
    def recorder(self, *, state=None, diagnostic=False):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        port = CameraPort(state)
        rec = preview.Recorder('/dev/never-open-real-device', Path(folder.name), diagnostic,
                               serial_factory=lambda: port, startup_s=0, interval_s=.005)
        self.addCleanup(rec.events.close)
        self.addCleanup(rec.serial_log.close)
        return rec, port

    def observe(self, rec, state):
        rec.lines(b'LAMP_MODEL '+json.dumps(state).encode()+b'\n')

    def test_formal_arm_rejects_startup_test_but_capture_thread_keeps_running(self):
        rec, port = self.recorder(state=model(armed=True), diagnostic=True)
        worker = threading.Thread(target=rec.run)
        worker.start()
        self.addCleanup(worker.join, 2)
        self.addCleanup(rec.stop.set)
        with port.condition:
            self.assertTrue(port.condition.wait_for(lambda: port.images >= 3, timeout=2))
        self.assertTrue(worker.is_alive())
        self.assertGreaterEqual(rec.frame, 2)
        self.assertFalse(rec.test_request.is_set())
        self.assertIn('ARM', rec.test_error)
        self.assertFalse(rec.error)
        self.assertNotIn(b'T\n', port.writes)
        self.assertNotIn(b'U\n', port.writes)
        self.assertFalse(port.closed)

    def test_manual_mode_does_not_require_model_selftest(self):
        rec, port = self.recorder()
        rec.stream = port
        self.observe(rec, model(selftest=False))
        self.assertTrue(rec.status()['test_ready'])
        self.assertTrue(rec.request_test()[0])
        self.assertTrue(rec.model_status())
        rec.process_requests(True)
        self.assertEqual(port.writes.count(b'T\n'), 1)

    def test_automatic_mode_waits_for_model_selftest(self):
        rec, port = self.recorder(state=model(manual=False, selftest=False), diagnostic=True)
        rec.stream = port
        self.assertTrue(rec.model_status())
        self.assertFalse(rec.status()['test_ready'])
        rec.process_requests(True)
        self.assertNotIn(b'T\n', port.writes)
        self.assertTrue(rec.test_request.is_set())
        port.state = model(manual=False, selftest=True)
        self.assertTrue(rec.model_status())
        rec.process_requests(True)
        self.assertEqual(port.writes.count(b'T\n'), 1)

    def test_formal_arm_cancels_startup_wait_without_a_delayed_test(self):
        rec, port = self.recorder(state=model(manual=False, selftest=False), diagnostic=True)
        rec.stream = port
        rec.model_status();rec.process_requests(True)
        self.assertTrue(rec.test_request.is_set())
        port.state = model(manual=False, armed=True, selftest=True)
        rec.model_status();rec.process_requests(True)
        self.assertFalse(rec.test_request.is_set())
        port.state = model(manual=False, selftest=True)
        rec.model_status();rec.process_requests(True)
        self.assertNotIn(b'T\n', port.writes)

    def test_automatic_operator_request_is_refused_before_selftest(self):
        rec, port = self.recorder()
        self.observe(rec, model(manual=False, selftest=False))
        self.assertFalse(rec.request_test()[0])
        self.assertFalse(rec.test_request.is_set())
        self.assertNotIn(b'T\n', port.writes)

    def test_queued_trial_is_dropped_when_maix_arms_and_never_runs_later(self):
        rec, port = self.recorder()
        rec.stream = port
        self.observe(rec, model())
        self.assertTrue(rec.request_test()[0])
        self.observe(rec, model(armed=True))
        rec.process_requests(True)
        self.assertFalse(rec.test_request.is_set())
        self.assertIn('ARM', rec.test_error)
        self.observe(rec, model())
        rec.process_requests(True)
        self.assertNotIn(b'T\n', port.writes)
        self.assertTrue(rec.image())
        self.assertTrue(rec.image())
        self.assertEqual(rec.frame, 2)

    def test_queued_trial_is_dropped_when_current_status_refresh_fails(self):
        rec, port = self.recorder()
        rec.stream = port
        self.observe(rec, model())
        self.assertTrue(rec.request_test()[0])
        rec.process_requests(False)
        self.observe(rec, model())
        rec.process_requests(True)
        self.assertNotIn(b'T\n', port.writes)
        self.assertFalse(rec.test_request.is_set())

    def test_changed_saved_configuration_invalidates_queued_trial(self):
        rec, port = self.recorder()
        rec.stream = port
        self.observe(rec, model())
        self.assertTrue(rec.request_test()[0])
        self.observe(rec, model(roi=[90, 40, 60, 60]))
        rec.process_requests(True)
        self.assertNotIn(b'T\n', port.writes)
        self.assertIn('范围已变化', rec.test_error)

    def test_saving_configuration_cancels_queued_trial(self):
        rec, port = self.recorder()
        rec.stream = port
        self.observe(rec, model())
        self.assertTrue(rec.request_test()[0])
        self.assertTrue(rec.request_configuration(dict(manual=False))[0])
        self.assertFalse(rec.test_request.is_set())
        self.assertFalse(rec.status()['test_ready'])
        rec.process_requests(True)
        self.assertEqual(port.writes, [b'R{"manual":false}\n'])
        self.assertNotIn(b'T\n', port.writes)

    def test_fresh_model_can_acknowledge_configuration_if_roi_reply_was_lost(self):
        rec, port = self.recorder()
        rec.stream = port
        self.observe(rec, model())
        self.assertTrue(rec.request_configuration(dict(manual=False))[0])
        rec.process_requests(True)
        self.assertIsNotNone(rec.inflight_configuration)
        self.observe(rec, model(manual=False))
        self.assertIsNone(rec.inflight_configuration)
        self.assertTrue(rec.status()['test_ready'])
        self.assertFalse(rec.config_error)

    def test_formal_arm_cancels_unacknowledged_configuration_without_retry(self):
        rec, port = self.recorder()
        rec.stream = port
        self.observe(rec, model())
        self.assertTrue(rec.request_configuration(dict(manual=False))[0])
        rec.process_requests(True)
        self.observe(rec, model(armed=True))
        self.assertIsNone(rec.inflight_configuration)
        self.assertIn('ARM', rec.config_error)
        self.observe(rec, model())
        rec.process_requests(True)
        self.assertEqual(port.writes, [b'R{"manual":false}\n'])
        self.assertTrue(rec.status()['test_ready'])

    def test_queued_configuration_is_not_sent_after_formal_arm(self):
        rec, port = self.recorder()
        rec.stream = port
        self.observe(rec, model())
        self.assertTrue(rec.request_configuration(dict(manual=False))[0])
        self.observe(rec, model(armed=True))
        rec.process_requests(True)
        self.assertEqual(port.writes, [])
        self.assertIn('ARM', rec.config_error)

    def test_formal_arm_clears_local_trial_ownership(self):
        rec, port = self.recorder()
        rec.stream, rec.test_started = port, True
        self.observe(rec, model(armed=True))
        self.assertFalse(rec.test_started)
        self.assertFalse(rec.status()['test_ready'])

    def test_missing_stale_or_invalid_diagnostics_cannot_start(self):
        rec, port = self.recorder()
        rec.stream = port
        self.assertFalse(rec.request_test()[0])
        for state in ({'diagnostic': None}, {'diagnostic': {}}, model(roi=[-1, 0, 60, 60])):
            with self.subTest(state=state):
                self.observe(rec, state)
                self.assertFalse(rec.request_test()[0])
        self.observe(rec, model())
        rec.model_at -= preview.MODEL_STATUS_MAX_AGE_S+1
        self.assertFalse(rec.status()['test_ready'])
        self.assertFalse(rec.request_test()[0])
        self.assertEqual(port.writes, [])

    def server(self, rec):
        server = preview.ThreadingHTTPServer(('127.0.0.1', 0), preview.make_handler(rec))
        worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
        worker.start()
        self.addCleanup(worker.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return 'http://127.0.0.1:%d' % server.server_address[1]

    def request(self, base, path, payload=None):
        request = Request(base+path, data=None if payload is None else json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'})
        try:
            response = build_opener(ProxyHandler({})).open(request, timeout=2)
        except HTTPError as exc:
            response = exc
        with response:
            return response.status, json.load(response)

    def test_http_formal_arm_returns_409_and_status_remains_available(self):
        rec, port = self.recorder()
        self.observe(rec, model(armed=True))
        base = self.server(rec)
        status, reply = self.request(base, '/test', {})
        self.assertEqual(status, 409)
        self.assertIn('ARM', reply['error'])
        self.assertFalse(reply['controls_start'])
        self.assertFalse(rec.test_request.is_set())
        status, reply = self.request(base, '/status')
        self.assertEqual(status, 200)
        self.assertFalse(reply['test_ready'])
        self.assertFalse(reply['control_ready'])
        self.assertEqual(reply['session_kind'], 'ISOLATED_USB_TEST')
        self.assertEqual(port.writes, [])

    def test_http_configuration_is_refused_during_formal_arm(self):
        rec, port = self.recorder()
        self.observe(rec, model(armed=True))
        base = self.server(rec)
        status, reply = self.request(base, '/roi', dict(manual=False))
        self.assertEqual(status, 409)
        self.assertIn('ARM', reply['error'])
        self.assertTrue(rec.commands.empty())
        self.assertEqual(port.writes, [])

    def test_http_manual_without_selftest_can_queue_but_automatic_cannot(self):
        rec, port = self.recorder()
        self.observe(rec, model(selftest=False))
        base = self.server(rec)
        status, reply = self.request(base, '/test', {})
        self.assertEqual(status, 200)
        self.assertTrue(reply['queued'])
        self.assertFalse(reply['controls_start'])
        rec.test_request.clear()
        self.observe(rec, model(manual=False, selftest=False))
        status, reply = self.request(base, '/test', {})
        self.assertEqual(status, 409)
        self.assertFalse(rec.test_request.is_set())
        self.assertEqual(port.writes, [])

    @unittest.skipUnless(NODE, 'Node.js unavailable; run browser checks with a JS runtime')
    def test_browser_controls_and_independent_session_messages(self):
        runner = "const fs=require('node:fs');const checks=fs.readFileSync(0,'utf8');new Function('require','source',checks)(require,fs.readFileSync(process.argv[1],'utf8')).then(console.log).catch(e=>{console.error(e);process.exitCode=1;});"
        subprocess.run([NODE, '-e', runner, str(preview.PAGE)], input=JS_CHECKS,
                       text=True, check=True, capture_output=True)

    def test_http_queued_request_cannot_cross_into_formal_session(self):
        rec, port = self.recorder()
        rec.stream = port
        self.observe(rec, model())
        base = self.server(rec)
        self.assertEqual(self.request(base, '/test', {})[0], 200)
        self.observe(rec, model(armed=True))
        rec.process_requests(True)
        self.assertEqual(self.request(base, '/status')[0], 200)
        self.assertFalse(rec.test_request.is_set())
        self.assertNotIn(b'T\n', port.writes)
        self.assertTrue(rec.image())
        self.assertTrue(rec.image())
        self.assertEqual(rec.frame, 2)


if __name__ == '__main__':
    unittest.main()
