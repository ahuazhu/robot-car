"""Exercise the actual ESP HTTP/USB configuration code with host-only adapters."""
import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from tests.support import ROOT

STUBS = r'''
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <new>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
class String {
 public:
  std::string value;
  String(){} String(const char *s):value(s?s:""){} String(const std::string&s):value(s){}
  template<class T> String(T n){std::ostringstream out;out<<n;value=out.str();}
  String(float n,int digits){std::ostringstream out;out<<std::fixed<<std::setprecision(digits)<<n;value=out.str();}
  size_t length()const{return value.size();}const char*c_str()const{return value.c_str();}
  char operator[](size_t i)const{return value[i];}
  String&operator+=(const String&s){value+=s.value;return *this;}
  bool operator==(const String&s)const{return value==s.value;}bool operator!=(const String&s)const{return value!=s.value;}
  friend String operator+(const String&a,const String&b){return a.value+b.value;}
};
typedef int* SemaphoreHandle_t;
static int lockDepth=0;static bool writeWasLocked=true;
#define portMAX_DELAY 0
static void xSemaphoreTake(SemaphoreHandle_t,int){++lockDepth;}
static void xSemaphoreGive(SemaphoreHandle_t){--lockDepth;}
static uint32_t millis(){return 1000;}
#define MALLOC_CAP_SPIRAM 1
#define MALLOC_CAP_8BIT 2
static void*heap_caps_malloc(size_t n,int){return malloc(n);}
static std::vector<uint8_t> savedBytes;static int storageFailure=0,writeCalls=0;
class Preferences{
 public:
  bool begin(const char*,bool){return storageFailure!=3;}void end(){}
  size_t putBytes(const char*,const void*data,size_t n){++writeCalls;writeWasLocked&=lockDepth>0;if(storageFailure==1)return 0;savedBytes.assign((const uint8_t*)data,(const uint8_t*)data+n);return n;}
  size_t getBytes(const char*,void*out,size_t n){size_t count=std::min(n,savedBytes.size());memcpy(out,savedBytes.data(),count);if(storageFailure==2&&count)((uint8_t*)out)[0]^=1;return count;}
  void putString(const char*,const String&){}String getString(const char*,const char*){return "";}
};
struct SerialStub{std::string last;void println(const String&s){last=s.value;}};static SerialStub Serial;
#define WL_CONNECTED 3
struct WiFiStub{int status(){return WL_CONNECTED;}struct IP{String toString(){return "10.0.0.2";}};IP localIP(){return {};}};static WiFiStub WiFi;
namespace lampnet{struct Statistics{int x=0,y=0,w=0,h=0;bool valid=false,selfTestOK=false;uint32_t sourceAt=0;};static Statistics testStats;static Statistics snapshot(){return testStats;}}
struct cJSON{int type;double valuedouble=0;int valueint=0;char*valuestring=nullptr;std::vector<std::pair<std::string,cJSON*>>object;std::vector<cJSON*>array;};
static bool cJSON_IsObject(const cJSON*v){return v&&v->type==1;}static bool cJSON_IsArray(const cJSON*v){return v&&v->type==2;}
static bool cJSON_IsBool(const cJSON*v){return v&&(v->type==3||v->type==4);}static bool cJSON_IsTrue(const cJSON*v){return v&&v->type==4;}
static bool cJSON_IsNumber(const cJSON*v){return v&&v->type==5;}static bool cJSON_IsString(const cJSON*v){return v&&v->type==6;}
static cJSON*cJSON_GetObjectItemCaseSensitive(const cJSON*v,const char*k){if(v)for(auto&p:v->object)if(p.first==k)return p.second;return nullptr;}
static int cJSON_GetArraySize(const cJSON*v){return v?int(v->array.size()):0;}static cJSON*cJSON_GetArrayItem(const cJSON*v,int i){return v&&i>=0&&i<int(v->array.size())?v->array[i]:nullptr;}
static void cJSON_Delete(cJSON*v){if(!v)return;for(auto&p:v->object)cJSON_Delete(p.second);for(auto*p:v->array)cJSON_Delete(p);free(v->valuestring);delete v;}
static cJSON*clone(const cJSON*v){if(!v)return nullptr;auto*c=new cJSON(*v);c->valuestring=v->valuestring?strdup(v->valuestring):nullptr;c->object.clear();c->array.clear();for(auto&p:v->object)c->object.push_back({p.first,clone(p.second)});for(auto*p:v->array)c->array.push_back(clone(p));return c;}
static cJSON* fixtureRoot=nullptr;static int duringParse=0;
static cJSON*cJSON_Parse(const char*);
typedef int esp_err_t;
#define ESP_OK 0
#define ESP_FAIL -1
#define HTTPD_403_FORBIDDEN 403
#define HTTPD_400_BAD_REQUEST 400
struct httpd_req_t{int content_len=0;std::string input,key,output,status="200 OK",contentType;size_t at=0;};
static int httpd_req_get_hdr_value_len(httpd_req_t*r,const char*){return r->key.size();}
static int httpd_req_get_hdr_value_str(httpd_req_t*r,const char*,char*out,size_t n){if(r->key.size()+1>n)return ESP_FAIL;memcpy(out,r->key.c_str(),r->key.size()+1);return ESP_OK;}
static int httpd_req_recv(httpd_req_t*r,char*out,size_t n){n=std::min(n,std::min(size_t(7),r->input.size()-r->at));memcpy(out,r->input.data()+r->at,n);r->at+=n;return n;}
static void httpd_resp_set_status(httpd_req_t*r,const char*s){r->status=s;}static void httpd_resp_set_type(httpd_req_t*r,const char*s){r->contentType=s;}
static void httpd_resp_set_hdr(httpd_req_t*,const char*,const char*){}
static int httpd_resp_send(httpd_req_t*r,const char*s,size_t n){r->output.assign(s,n);return ESP_OK;}
static int httpd_resp_sendstr(httpd_req_t*r,const char*s){return httpd_resp_send(r,s,strlen(s));}
static int httpd_resp_send_err(httpd_req_t*r,int status,const char*s){r->status=std::to_string(status);return httpd_resp_sendstr(r,s);}
'''

BRIDGE = r'''
static cJSON*cJSON_Parse(const char*){
  if(duringParse==1)signalDetector.armed=true;
  if(duringParse==2)signalKey=std::string(64,'b');
  if(duringParse==3)strcpy(signalBoot,"fedcba9876543210");
  return clone(fixtureRoot);
}
static httpd_req_t lastRequest;static std::string scratch;
extern "C" {
void reset(int armed,int confirmed,int diagnostic,int failure,int race){
  signalDetector.cancel();signalDetector.armed=armed;signalDetector.confirmed=confirmed;signalDetector.sequence=20;
  signalKey=std::string(64,'a');strcpy(signalBoot,"0123456789abcdef");signalSession="";lastSignalSession="";signalLease=99;
  manualLampEnabled=true;int roi[4]={139,51,63,61};memcpy(manualLampROI,roi,sizeof(roi));
  if(modelTestDetector)delete modelTestDetector;modelTestDetector=new start_signal::Detector();modelTestDetector->arm();modelTestEnabled=diagnostic;
  int initial[5]={1,139,51,63,61};savedBytes.assign((uint8_t*)initial,(uint8_t*)initial+sizeof(initial));
  lampnet::testStats=lampnet::Statistics();storageFailure=failure;duringParse=race;writeCalls=0;writeWasLocked=true;lockDepth=0;Serial.last.clear();
}
cJSON*node(int type,double value,const char*text){auto*p=new cJSON();p->type=type;p->valuedouble=value;p->valueint=value>INT_MAX?INT_MAX:value<INT_MIN?INT_MIN:int(value);if(type==6)p->valuestring=strdup(text?text:"");return p;}
void member(cJSON*p,const char*k,cJSON*v){p->object.push_back({k,v});}void append(cJSON*p,cJSON*v){p->array.push_back(v);}void destroyNode(cJSON*p){cJSON_Delete(p);}
int request(cJSON*root,const char*body,const char*key){fixtureRoot=root;lastRequest=httpd_req_t();lastRequest.input=body;lastRequest.content_len=strlen(body);lastRequest.key=key;signalControl(&lastRequest);return atoi(lastRequest.status.c_str());}
const char*response(){return lastRequest.output.c_str();}
const char*status(){scratch=signalJSON().value;return scratch.c_str();}
void usb(cJSON*root){fixtureRoot=root;configureLampROI("fixture");}
const char*serialMessage(){return Serial.last.c_str();}
void hint(int valid,int x,int y,int w,int h,unsigned source,int selftest){auto&s=lampnet::testStats;s.valid=valid;s.x=x;s.y=y;s.w=w;s.h=h;s.sourceAt=source;s.selfTestOK=selftest;}
int writes(){return writeCalls;}int lockedWrite(){return writeWasLocked&&lockDepth==0;}int diagnosticEnabled(){return modelTestEnabled;}
int diagnosticArmed(){return modelTestDetector->armed;}int lease(){return signalLease;}int nativeArmed(){return signalDetector.armed;}
}
'''


class LampConfigurationHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(prefix='esp-lamp-configuration-')
        folder = Path(cls.folder.name)
        # Compile the production request/configuration prefix unchanged. Camera
        # capture/task functions below it are unrelated to this host-only test.
        source = Path(__file__).resolve().parents[1]/'esp32s3/video_preview/signal_service.h'
        prefix = source.read_text().split('static void cameraProducer(', 1)[0]
        cpp = folder/'http_configuration.cpp'
        cpp.write_text('#define ROBOT_LAMPNET_FULL_FRAME 1\n#define ROBOT_LAMPNET_FACE 1\n'+STUBS+'\n'+prefix+'\n'+BRIDGE)
        library = folder/'http_configuration.so'
        subprocess.run(['c++', '-std=c++11', '-O2', '-shared', '-fPIC', '-I', str(ROOT/'esp32s3/video_preview'), str(cpp), '-o', str(library)], check=True, capture_output=True)
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.reset.argtypes = [ctypes.c_int]*5
        cls.lib.node.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_char_p]
        cls.lib.node.restype = ctypes.c_void_p
        cls.lib.member.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        cls.lib.append.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cls.lib.destroyNode.argtypes = [ctypes.c_void_p]
        cls.lib.request.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        cls.lib.usb.argtypes = [ctypes.c_void_p]
        cls.lib.hint.argtypes = [ctypes.c_int]*5+[ctypes.c_uint, ctypes.c_int]
        for name in ('response', 'status', 'serialMessage'):
            getattr(cls.lib, name).restype = ctypes.c_char_p

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def setUp(self):
        self.lib.reset(0, 0, 1, 0, 0)

    def tree(self, value):
        if isinstance(value, dict):
            ptr = self.lib.node(1, 0, None)
            for key, child in value.items():
                self.lib.member(ptr, key.encode(), self.tree(child))
        elif isinstance(value, list):
            ptr = self.lib.node(2, 0, None)
            for child in value:
                self.lib.append(ptr, self.tree(child))
        elif isinstance(value, bool):
            ptr = self.lib.node(4 if value else 3, 0, None)
        elif isinstance(value, (int, float)):
            ptr = self.lib.node(5, value, None)
        elif isinstance(value, str):
            ptr = self.lib.node(6, 0, value.encode())
        else:
            ptr = self.lib.node(0, 0, None)
        return ptr

    def configure(self, configuration, *, key='a'*64, boot='0123456789abcdef', **extra):
        payload = dict(op='configure', boot=boot, configuration=configuration, **extra)
        return self.request(payload, key)

    def request(self, payload, key='a'*64):
        root = self.tree(payload)
        try:
            code = self.lib.request(root, json.dumps(payload).encode(), key.encode())
            raw = self.lib.response().decode()
            return code, json.loads(raw) if raw.startswith('{') else raw
        finally:
            self.lib.destroyNode(root)

    def status(self):
        return json.loads(self.lib.status())

    def test_status_exposes_saved_manual_range_without_mutating_storage(self):
        self.assertEqual(self.status()['lamp_configuration'], dict(manual=True, roi=[139, 51, 63, 61]))
        self.assertEqual(self.lib.writes(), 0)

    def test_candidate_hint_is_only_a_fresh_bounded_snapshot(self):
        self.lib.hint(1, 120, 40, 64, 64, 900, 1)
        self.assertEqual(self.status()['candidate_hint'], dict(valid=True, roi=[120, 40, 64, 64], age_ms=100))
        self.lib.hint(1, 120, 40, 64, 64, (1000-2500)&0xffffffff, 1)
        self.assertTrue(self.status()['candidate_hint']['valid'])
        self.assertEqual(self.lib.writes(), 0)
        self.assertFalse(self.lib.nativeArmed())

    def test_stale_untrusted_or_out_of_bounds_candidates_are_not_drawn(self):
        for valid, x, y, w, h, source, selftest in (
                (0, 120, 40, 64, 64, 900, 1), (1, 120, 40, 64, 64, 900, 0),
                (1, 120, 40, 64, 64, 0, 1), (1, 120, 40, 64, 64, (1000-2501)&0xffffffff, 1),
                (1, -1, 40, 64, 64, 900, 1), (1, 300, 40, 64, 64, 900, 1),
                (1, 120, 40, 15, 64, 900, 1), (1, 120, 220, 64, 64, 900, 1)):
            self.lib.hint(valid, x, y, w, h, source, selftest)
            hint = self.status()['candidate_hint']
            self.assertFalse(hint['valid']);self.assertEqual(hint['roi'], [0, 0, 0, 0])
        self.assertEqual(self.lib.writes(), 0)

    def test_authenticated_manual_save_is_atomic_and_cancels_only_diagnostic(self):
        code, result = self.configure(dict(manual=True, roi=[60, 40, 80, 64]))
        self.assertEqual(code, 200)
        self.assertEqual(result['lamp_configuration'], dict(manual=True, roi=[60, 40, 80, 64]))
        self.assertFalse(result['armed']);self.assertFalse(result['confirmed']);self.assertEqual(result['session'], '')
        self.assertEqual(self.lib.lease(), 99)
        self.assertFalse(self.lib.diagnosticEnabled());self.assertFalse(self.lib.diagnosticArmed())
        self.assertEqual(self.lib.writes(), 1);self.assertTrue(self.lib.lockedWrite())
        self.assertNotIn('a'*64, json.dumps(result))

    def test_auto_mode_keeps_last_manual_rectangle(self):
        code, result = self.configure(dict(manual=False))
        self.assertEqual(code, 200)
        self.assertEqual(result['lamp_configuration'], dict(manual=False, roi=[139, 51, 63, 61]))

    def test_manual_bounds_and_types_are_checked_before_persistence(self):
        invalid = [None, {}, {'manual': 1}, {'manual': True}, {'manual': True, 'roi': [-1, 0, 60, 60]},
                   {'manual': True, 'roi': [0, 0, 321, 40]}, {'manual': True, 'roi': [0, 0, 12, 40]},
                   {'manual': True, 'roi': [True, 0, 40, 40]}, {'manual': True, 'roi': [1.5, 0, 40, 40]},
                   {'manual': True, 'roi': [2147483647, 0, 40, 40]}, {'manual': True, 'roi': [0, 0, 40]}]
        for configuration in invalid:
            with self.subTest(configuration=configuration):
                code, result = self.configure(configuration)
                self.assertEqual((code, result), (400, {'error': 'invalid_configuration'}))
                self.assertEqual(self.lib.writes(), 0)
                self.assertTrue(self.lib.diagnosticEnabled())

    def test_wrong_or_missing_key_cannot_write_or_disclose_configuration(self):
        for key in ('', 'b'*64, 'a'*63):
            code, result = self.configure(dict(manual=False), key=key)
            self.assertEqual((code, result), (403, {'error': 'pairing_required'}))
            self.assertEqual(self.lib.writes(), 0)

    def test_configuration_envelope_requires_a_boot_string_and_bounded_body(self):
        for boot in (None, True, 123):
            code, result = self.configure(dict(manual=False), boot=boot)
            self.assertEqual((code, result), (400, {'error': 'invalid_request'}))
        code, result = self.configure(dict(manual=False), padding='x'*200)
        self.assertEqual((code, result), (400, {'error': 'invalid_request'}))
        self.assertEqual(self.lib.writes(), 0)

    def test_stale_boot_is_rejected_without_configuration_or_session_changes(self):
        code, result = self.configure(dict(manual=False), boot='fedcba9876543210')
        self.assertEqual((code, result), (409, {'error': 'boot_changed'}))
        self.assertEqual(self.lib.writes(), 0)
        self.assertEqual(self.lib.lease(), 99)

    def test_formal_arm_and_confirmed_event_both_forbid_edits(self):
        for confirmed in (0, 1):
            self.lib.reset(1, confirmed, 0, 0, 0)
            code, result = self.configure(dict(manual=False))
            self.assertEqual((code, result), (409, {'error': 'native_armed_stop_first'}))
            self.assertEqual(self.lib.writes(), 0)
            self.assertTrue(self.status()['armed']);self.assertEqual(self.status()['confirmed'], bool(confirmed))

    def test_auth_boot_and_arm_are_rechecked_inside_persistence_transaction(self):
        for race, expected in ((1, (409, 'native_armed_stop_first')), (2, (403, 'pairing_required')), (3, (409, 'boot_changed'))):
            self.lib.reset(0, 0, 1, 0, race)
            code, result = self.configure(dict(manual=False))
            self.assertEqual((code, result['error']), expected)
            self.assertEqual(self.lib.writes(), 0)
            self.assertTrue(self.lib.diagnosticEnabled())

    def test_storage_failure_does_not_change_live_range_or_cancel_diagnostic(self):
        for failure in (1, 2, 3):
            self.lib.reset(0, 0, 1, failure, 0)
            code, result = self.configure(dict(manual=False))
            self.assertEqual((code, result), (500, {'error': 'storage_failed'}))
            self.assertEqual(self.status()['lamp_configuration'], dict(manual=True, roi=[139, 51, 63, 61]))
            self.assertTrue(self.lib.diagnosticEnabled());self.assertTrue(self.lib.diagnosticArmed())

    def test_usb_and_http_share_validation_and_saved_configuration(self):
        root = self.tree(dict(manual=True, roi=[120, 60, 64, 64]))
        try:self.lib.usb(root)
        finally:self.lib.destroyNode(root)
        self.assertTrue(self.lib.serialMessage().decode().startswith('ROI OK '))
        self.assertEqual(self.status()['lamp_configuration'], dict(manual=True, roi=[120, 60, 64, 64]))
        self.assertEqual(self.lib.writes(), 1)

    def test_existing_arm_poll_cancel_protocol_remains_compatible(self):
        session = '1'*32
        for operation in ('arm', 'poll', 'cancel'):
            code, result = self.request(dict(op=operation, session=session, boot='0123456789abcdef'))
            self.assertEqual(code, 200)
            self.assertEqual(result['armed'], operation != 'cancel')
        self.assertEqual(self.lib.writes(), 0)


if __name__ == '__main__':
    unittest.main()
