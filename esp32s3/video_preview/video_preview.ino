#include <Arduino.h>
#include <WiFi.h>
#include <Preferences.h>
#include <atomic>
#include <ESPmDNS.h>
#include "cJSON.h"
#include "esp_camera.h"
#include "esp_timer.h"
#include "esp_task_wdt.h"
#include "esp_http_server.h"
#include "img_converters.h"

// Hiwonder ESP32-S3 / GC2145, verified against manufacturer's camera_setting.h.
// ESP32 owns start-light sensing only; Maix remains the sole motor/start authority.
static bool cameraOK=false;
static uint32_t frames=0;
static std::atomic<bool> loopWatchdogReady(false),cameraWatchdogReady(false);
static bool networkConfigured=false;
static uint32_t lastReconnectAt=0,networkReconnects=0;
static httpd_handle_t web=nullptr, controlServer=nullptr;

#ifdef ROBOT_LAMPNET_SHADOW
#include "lampnet_shadow.h"
#endif
#include "signal_service.h"

#if defined(ROBOT_LAMPNET_FULL_FRAME)
static const char *FIRMWARE_VERSION="0.4.17-strong-rgb-or";
#elif defined(ROBOT_LAMPNET_FACE)
static const char *FIRMWARE_VERSION="0.4.9-lamp-face";
#elif defined(ROBOT_LAMPNET_SHADOW) && defined(ROBOT_LAMPNET_ESP_NN)
static const char *FIRMWARE_VERSION="0.4.8-lampnet-espnn";
#elif defined(ROBOT_LAMPNET_SHADOW)
static const char *FIRMWARE_VERSION="0.4.7-lampnet-probe";
#else
static const char *FIRMWARE_VERSION="0.4.6-larger-roi";
#endif
static String pendingSSID, pendingPassword;
static bool networkPending=false;
static std::atomic<bool> networkGotIP(false);
static uint32_t networkBegan=0;
static const char *networkError="";
static char configLine[768];
static size_t configUsed=0;
static bool readingConfig=false, configOverflow=false;
static uint32_t configBegan=0;
static char configCommand='C';

static void printNetworkStatus(){
  Serial.printf("WIFI {\"firmware\":\"%s\",\"connected\":%s,\"pending\":%s,\"ip\":\"%s\",\"error\":\"%s\"}\n",
    FIRMWARE_VERSION,WiFi.status()==WL_CONNECTED?"true":"false",networkPending?"true":"false",
    WiFi.status()==WL_CONNECTED?WiFi.localIP().toString().c_str():"",networkError);
}
static bool validCredentials(const char *ssid,const char *password){
  size_t n=strlen(ssid),m=strlen(password);
  if(n==0 || n>32 || (m!=0 && (m<8 || m>64)))return false;
  if(m==64){for(size_t i=0;i<m;++i)if(!isxdigit((unsigned char)password[i]))return false;}
  for(size_t i=0;i<n;++i)if((unsigned char)ssid[i]<32)return false;
  return true;
}
static void configureNetwork(const char *data){
  cJSON *root=cJSON_Parse(data);
  const cJSON *ssid=root?cJSON_GetObjectItemCaseSensitive(root,"ssid"):nullptr;
  const cJSON *password=root?cJSON_GetObjectItemCaseSensitive(root,"password"):nullptr;
  if(!cJSON_IsString(ssid) || !cJSON_IsString(password) || !validCredentials(ssid->valuestring,password->valuestring)){
    if(root)cJSON_Delete(root);
    Serial.println("CONFIG ERROR invalid_credentials");return;
  }
  pendingSSID=ssid->valuestring;pendingPassword=password->valuestring;
  cJSON_Delete(root);
  networkPending=false;
  WiFi.disconnect(false,false);
  uint32_t disconnectBegan=millis();
  while(WiFi.status()==WL_CONNECTED && uint32_t(millis()-disconnectBegan)<1000)delay(10);
  if(WiFi.status()==WL_CONNECTED){
    pendingSSID="";pendingPassword="";Serial.println("CONFIG ERROR disconnect_failed");return;
  }
  networkGotIP.store(false);networkError="";networkPending=true;networkBegan=millis();
  WiFi.begin(pendingSSID.c_str(),pendingPassword.c_str());
  Serial.println("CONFIG ACCEPTED");
}
static void updateNetwork(){
  if(networkPending && networkGotIP.load() && WiFi.status()==WL_CONNECTED && WiFi.SSID()==pendingSSID){
    Preferences saved;
    bool stored=false;
    if(saved.begin("preview_wifi",false)){
      saved.putBool("configured",false);
      saved.putString("ssid",pendingSSID);saved.putString("password",pendingPassword);
      stored=saved.getString("ssid","")==pendingSSID && saved.getString("password","")==pendingPassword;
      saved.putBool("configured",stored);saved.end();
    }
    if(stored)networkConfigured=true;
    pendingSSID="";pendingPassword="";networkPending=false;networkError=stored?"":"storage_failed";
    printNetworkStatus();
  }else if(networkPending && uint32_t(millis()-networkBegan)>25000){
    pendingSSID="";pendingPassword="";networkPending=false;networkError="connection_timeout";
    printNetworkStatus();
  }
}
static void startNetwork(){
  WiFi.persistent(false);
  WiFi.onEvent([](WiFiEvent_t event){
    if(event==ARDUINO_EVENT_WIFI_STA_GOT_IP)networkGotIP.store(true);
    if(event==ARDUINO_EVENT_WIFI_STA_DISCONNECTED)networkGotIP.store(false);
  });
  WiFi.mode(WIFI_AP_STA);WiFi.setSleep(false);WiFi.setAutoReconnect(true);
  WiFi.setHostname("robot-esp32cam");
  WiFi.softAPConfig(IPAddress(192,168,5,1),IPAddress(192,168,5,1),IPAddress(255,255,255,0));
  if(!WiFi.softAP("HW_ESP32S3CAM"))Serial.println("ERROR WiFi AP");
  Preferences saved;
  if(saved.begin("preview_wifi",true)){
    String ssid=saved.getBool("configured",false)?saved.getString("ssid",""):"",password=saved.getString("password","");saved.end();
    networkConfigured=ssid.length()>0;
    if(networkConfigured)WiFi.begin(ssid.c_str(),password.c_str());
  }
  if(MDNS.begin("robot-esp32cam"))MDNS.addService("http","tcp",80);
}
static const char PAGE[] PROGMEM=R"HTML(<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ESP32-S3 近场摄像头</title><style>body{background:#111827;color:#e5e7eb;font:17px system-ui;margin:20px auto;max-width:900px;padding:0 15px}button,a{font:inherit;padding:10px;margin:4px}#view{position:relative;width:100%;aspect-ratio:4/3;background:#000}img{width:100%;height:100%;object-fit:contain}#grid{position:absolute;inset:0;pointer-events:none;background:linear-gradient(90deg,transparent 49.7%,#38bdf8 50%,transparent 50.3%),linear-gradient(transparent 49.7%,#38bdf8 50%,transparent 50.3%);border:1px solid #38bdf8}small{color:#94a3b8}</style><h2>ESP32-S3 近场摄像头</h2><p>用于调整安装位置和仰角。让龙门架信号灯完整可见。灯灭时框好灯面并按 Maix ARM，等待亮灯后再次熄灭；ESP32 不直接控制电机。</p><div id="view"><img id="cam"><div id="searchROI" hidden style="position:absolute;left:25%;top:0%;width:50%;height:53.333333%;border:2px solid cyan;box-sizing:border-box;pointer-events:none"></div><div id="target" hidden style="position:absolute;border:2px solid #facc15;box-sizing:border-box;pointer-events:none"></div><div id="grid" hidden></div></div><p><button onclick="start()">开始预览</button><button onclick="stop()">停止预览</button><button onclick="let g=document.getElementById('grid');g.hidden=!g.hidden">中心参考线</button></p><small>320 × 240 · 模型搜索整个画面；只检测锁定的灯面。黄色小框锁定灯面后，连续至少3帧、0.4秒确认熄灭。</small><script>let running=false,timer=null,deadline=null,request=0;const image=document.getElementById('cam');
function stop(){running=false;request++;clearTimeout(timer);clearTimeout(deadline);image.onload=image.onerror=null;image.removeAttribute('src');}
function frame(){if(!running||document.hidden)return;const id=++request;function retry(){if(id!==request)return;clearTimeout(deadline);image.onload=image.onerror=null;image.removeAttribute('src');timer=setTimeout(frame,500);}image.onload=()=>{if(id!==request)return;clearTimeout(deadline);timer=setTimeout(frame,150);};image.onerror=retry;deadline=setTimeout(retry,1500);image.src='/capture?t='+Date.now();}
function start(){stop();running=true;frame();}document.addEventListener('visibilitychange',()=>{if(document.hidden){const resume=running;stop();running=resume;}else if(running)frame();});
async function status(){if(document.hidden){setTimeout(status,500);return;}try{const r=await fetch('/status',{cache:'no-store',signal:AbortSignal.timeout(1500)});if(!r.ok)throw Error();const s=(await r.json()).signal,t=document.getElementById('target'),a=s.target_roi;document.getElementById('searchROI').hidden=s.roi[2]===320; t.hidden=!s.target_locked;if(!t.hidden)Object.assign(t.style,{left:a[0]/320*100+'%',top:a[1]/240*100+'%',width:a[2]/320*100+'%',height:a[3]/240*100+'%'});}catch(e){document.getElementById('target').hidden=true;}setTimeout(status,500);}status();start()</script></html>)HTML";

static bool getJPEG(uint8_t **jpg,size_t *size){
  if(!cameraOK)return false;
  xSemaphoreTake(jpegLock,portMAX_DELAY);
  bool ok=latestJPEG && uint32_t(millis()-latestJPEGAt)<=500;
  if(ok){*size=latestJPEGSize;*jpg=(uint8_t*)malloc(*size);ok=*jpg!=nullptr;if(ok)memcpy(*jpg,latestJPEG,*size);}
  xSemaphoreGive(jpegLock);return ok;
}
static esp_err_t indexHandler(httpd_req_t *r){
  httpd_resp_set_type(r,"text/html; charset=utf-8");
  return httpd_resp_send(r,PAGE,HTTPD_RESP_USE_STRLEN);
}
static esp_err_t captureHandler(httpd_req_t *r){
  uint8_t *jpg=nullptr;size_t size=0;
  if(!getJPEG(&jpg,&size))return httpd_resp_send_err(r,HTTPD_500_INTERNAL_SERVER_ERROR,"Camera capture failed");
  httpd_resp_set_type(r,"image/jpeg");httpd_resp_set_hdr(r,"Cache-Control","no-store");
  esp_err_t res=httpd_resp_send(r,(const char*)jpg,size);free(jpg);return res;
}
static esp_err_t statusHandler(httpd_req_t *r){
  String data="{\"firmware\":\""+String(FIRMWARE_VERSION)+"\",\"camera\":"+(cameraOK?"true":"false")+",\"frames\":"+String(frames)+",\"station_connected\":"+(WiFi.status()==WL_CONNECTED?"true":"false")+",\"ip\":\""+(WiFi.status()==WL_CONNECTED?WiFi.localIP().toString():String(""))+"\",\"signal\":"+signalJSON()+"}";
  data.remove(data.length()-1);
  data+=",\"diagnostics\":{\"uptime_ms\":"+String(millis())+",\"reset_reason\":"+String(int(esp_reset_reason()))+",\"free_heap\":"+String(ESP.getFreeHeap())+",\"wifi_rssi\":"+String(WiFi.RSSI())+",\"network_reconnects\":"+String(networkReconnects)+",\"loop_watchdog\":"+String(loopWatchdogReady.load()?"true":"false")+",\"camera_watchdog\":"+String(cameraWatchdogReady.load()?"true":"false")+",\"watchdog_timeout_s\":8}}";
#ifdef ROBOT_LAMPNET_SHADOW
  data.remove(data.length()-1);data+=",\"lampnet\":"+lampnet::statusJSON()+"}";
#endif
  httpd_resp_set_type(r,"application/json");httpd_resp_set_hdr(r,"Cache-Control","no-store");
  return httpd_resp_send(r,data.c_str(),data.length());
}
static bool startServers(){
  httpd_config_t conf=HTTPD_DEFAULT_CONFIG();conf.stack_size=8192;conf.recv_wait_timeout=1;conf.send_wait_timeout=1;conf.max_open_sockets=3;conf.lru_purge_enable=true;
  if(httpd_start(&web,&conf)!=ESP_OK)return false;
  httpd_uri_t root={};root.uri="/";root.method=HTTP_GET;root.handler=indexHandler;
  httpd_uri_t capture={};capture.uri="/capture";capture.method=HTTP_GET;capture.handler=captureHandler;
  httpd_register_uri_handler(web,&root);httpd_register_uri_handler(web,&capture);
  httpd_uri_t status={};status.uri="/status";status.method=HTTP_GET;status.handler=statusHandler;
  httpd_register_uri_handler(web,&status);
  // Keep small authenticated control traffic out of the JPEG send queue.
  conf.server_port=81;conf.ctrl_port+=1;conf.max_open_sockets=2;conf.task_priority+=1;
  if(httpd_start(&controlServer,&conf)!=ESP_OK)return false;
  httpd_register_uri_handler(controlServer,&status);
  httpd_uri_t control={};control.uri="/signal";control.method=HTTP_POST;control.handler=signalControl;
  return httpd_register_uri_handler(controlServer,&control)==ESP_OK;
}
void setup(){
  Serial.begin(115200);Serial.setDebugOutput(false);delay(500);
  Serial.println("ESP32-S3 PREVIEW | motors not controlled | I2C lane data disabled");
  initSignal();
  if(!signalLock || !jpegLock || !psramFound()){Serial.println("ERROR PSRAM/mutex unavailable");return;}
  if(esp_task_wdt_init(8,true)==ESP_OK)loopWatchdogReady.store(esp_task_wdt_status(nullptr)==ESP_OK||esp_task_wdt_add(nullptr)==ESP_OK);
  camera_config_t c={};
  c.ledc_channel=LEDC_CHANNEL_0;c.ledc_timer=LEDC_TIMER_0;
  c.pin_d0=11;c.pin_d1=9;c.pin_d2=8;c.pin_d3=10;c.pin_d4=12;c.pin_d5=18;c.pin_d6=17;c.pin_d7=16;
  c.pin_xclk=15;c.pin_pclk=13;c.pin_vsync=6;c.pin_href=7;
  c.pin_sccb_sda=4;c.pin_sccb_scl=5;c.pin_pwdn=-1;c.pin_reset=-1;
  c.xclk_freq_hz=15000000;c.pixel_format=PIXFORMAT_RGB565;c.frame_size=FRAMESIZE_QVGA;
  c.jpeg_quality=16;c.fb_count=1;c.fb_location=CAMERA_FB_IN_PSRAM;c.grab_mode=CAMERA_GRAB_WHEN_EMPTY;
  esp_err_t err=esp_camera_init(&c);
  if(err!=ESP_OK){Serial.printf("ERROR camera init 0x%x\n",err);return;}
  cameraOK=true;
  sensor_t *s=esp_camera_sensor_get();Serial.printf("CAMERA OK PID=0x%x PSRAM=%u\n",s->id.PID,unsigned(ESP.getPsramSize()));
  if(xTaskCreatePinnedToCore(cameraProducer,"camera_signal",16384,nullptr,2,nullptr,1)!=pdPASS){Serial.println("ERROR camera task");return;}
  startNetwork();
#ifdef ROBOT_LAMPNET_SHADOW
  lampnet::init();
#endif
  if(!startServers()){Serial.println("ERROR HTTP server");return;}
  Serial.println("PREVIEW READY | WiFi=HW_ESP32S3CAM | http://192.168.5.1/ | preview=80 control=81");
  Serial.println("USB commands: J=snapshot JPEG, S=status, Q=network status, C+JSON=Wi-Fi setup");
}
void loop(){
  if(loopWatchdogReady.load())esp_task_wdt_reset();
  updateNetwork();
  if(networkConfigured&&!networkPending&&WiFi.status()!=WL_CONNECTED&&uint32_t(millis()-lastReconnectAt)>=5000){
    lastReconnectAt=millis();++networkReconnects;WiFi.reconnect();
  }
  if(cameraOK&&millis()>5000&&uint32_t(millis()-signalFrameAt)>3000){
    Serial.println("RECOVERY camera_frames_stalled");Serial.flush();ESP.restart();
  }
  xSemaphoreTake(signalLock,portMAX_DELAY);
  if(signalDetector.armed&&!signalDetector.confirmed&&uint32_t(millis()-signalLease)>2000)cancelSignal();
  xSemaphoreGive(signalLock);
  if(readingConfig && uint32_t(millis()-configBegan)>2000){
    readingConfig=false;configUsed=0;Serial.println("CONFIG ERROR incomplete_request");
  }
  while(Serial.available()){
    char cmd=Serial.read();
    if(readingConfig){
      if(cmd=='\n'){
        configLine[configUsed]='\0';readingConfig=false;
        if(configOverflow)Serial.println("CONFIG ERROR request_too_large");
#ifdef ROBOT_LAMPNET_FACE
        else if(configCommand=='R')configureLampROI(configLine);
#endif
        else if(configCommand=='P')pairSignal(configLine);else configureNetwork(configLine);
        memset(configLine,0,sizeof(configLine));configUsed=0;
      }else if(cmd!='\r'){
        if(configUsed<sizeof(configLine)-1)configLine[configUsed++]=cmd;else configOverflow=true;
      }
      continue;
    }
    if(cmd=='C'||cmd=='P'
#ifdef ROBOT_LAMPNET_FACE
       ||cmd=='R'
#endif
    ){configCommand=cmd;readingConfig=true;configOverflow=false;configUsed=0;configBegan=millis();}
#ifdef ROBOT_LAMPNET_SHADOW
    if(cmd=='M'){
      String model=lampnet::statusJSON();
#ifdef ROBOT_LAMPNET_FACE
      model.remove(model.length()-1);model+=",\"diagnostic\":"+modelTestJSON()+"}";
#endif
      Serial.println("LAMP_MODEL "+model);
    }
#ifdef ROBOT_LAMPNET_FACE
    if(cmd=='T')modelTestCommand(true);
    if(cmd=='U')modelTestCommand(false);
#endif
#endif
    if(cmd=='Q')printNetworkStatus();
    if(cmd=='S')Serial.printf("STATUS camera=%u clients=%u frames=%u sta=%u\n",cameraOK,WiFi.softAPgetStationNum(),unsigned(frames),WiFi.status()==WL_CONNECTED);
    if(cmd=='J'){
      uint8_t *jpg=nullptr;size_t size=0;
      if(getJPEG(&jpg,&size)){
        Serial.printf("JPEG %u\n",unsigned(size));for(size_t at=0;at<size;at+=256){Serial.write(jpg+at,size-at<256?size-at:256);if(loopWatchdogReady.load())esp_task_wdt_reset();}Serial.flush();free(jpg);
      }else Serial.println("ERROR snapshot");
    }
  }
  delay(5);
}
