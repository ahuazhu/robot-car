#include <Arduino.h>
#include <WiFi.h>
#include "esp_camera.h"
#include "esp_http_server.h"
#include "img_converters.h"

// Hiwonder ESP32-S3 / GC2145, verified against manufacturer's camera_setting.h.
// Preview only: no motor commands and no external I2C-controller interface.
static SemaphoreHandle_t cameraLock;
static bool cameraOK=false;
static uint32_t frames=0;
static httpd_handle_t web=nullptr, stream=nullptr;
static const char PAGE[] PROGMEM=R"HTML(<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ESP32-S3 近场摄像头</title><style>body{background:#111827;color:#e5e7eb;font:17px system-ui;margin:20px auto;max-width:900px;padding:0 15px}button,a{font:inherit;padding:10px;margin:4px}#view{position:relative;width:100%;aspect-ratio:4/3;background:#000}img{width:100%;height:100%;object-fit:contain}#grid{position:absolute;inset:0;pointer-events:none;background:linear-gradient(90deg,transparent 49.7%,#38bdf8 50%,transparent 50.3%),linear-gradient(transparent 49.7%,#38bdf8 50%,transparent 50.3%);border:1px solid #38bdf8}small{color:#94a3b8}</style><h2>ESP32-S3 近场摄像头</h2><p>用于调整安装位置和俯角。当前仅预览，不控制小车。</p><div id="view"><img id="cam"><div id="grid"></div></div><p><button onclick="start()">开始预览</button><button onclick="document.getElementById('cam').removeAttribute('src')">停止预览</button><button onclick="let g=document.getElementById('grid');g.hidden=!g.hidden">中心参考线</button></p><small>320 × 240 · 观察左右白边能否同时进入画面，并检查弯道视野。参考线不是识别结果。</small><script>function start(){document.getElementById('cam').src='http://'+location.hostname+':81/stream?t='+Date.now()}start()</script></html>)HTML";

static bool getJPEG(uint8_t **jpg,size_t *size){
  if(!cameraOK || xSemaphoreTake(cameraLock,pdMS_TO_TICKS(2000))!=pdTRUE)return false;
  camera_fb_t *fb=esp_camera_fb_get();
  bool ok=fb && frame2jpg(fb,70,jpg,size);
  if(fb)esp_camera_fb_return(fb);
  xSemaphoreGive(cameraLock);
  if(ok)++frames;
  return ok;
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
static esp_err_t streamHandler(httpd_req_t *r){
  httpd_resp_set_type(r,"multipart/x-mixed-replace;boundary=frame");
  httpd_resp_set_hdr(r,"Cache-Control","no-store");
  esp_err_t res=ESP_OK;
  while(res==ESP_OK){
    uint8_t *jpg=nullptr;size_t size=0;
    if(!getJPEG(&jpg,&size))break;
    char header[100];int len=snprintf(header,sizeof(header),"\r\n--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",unsigned(size));
    res=httpd_resp_send_chunk(r,header,len);
    if(res==ESP_OK)res=httpd_resp_send_chunk(r,(const char*)jpg,size);
    free(jpg);vTaskDelay(pdMS_TO_TICKS(60));
  }
  return ESP_FAIL;
}
static bool startServers(){
  httpd_config_t conf=HTTPD_DEFAULT_CONFIG();conf.stack_size=8192;conf.recv_wait_timeout=5;conf.send_wait_timeout=3;
  if(httpd_start(&web,&conf)!=ESP_OK)return false;
  httpd_uri_t root={};root.uri="/";root.method=HTTP_GET;root.handler=indexHandler;
  httpd_uri_t capture={};capture.uri="/capture";capture.method=HTTP_GET;capture.handler=captureHandler;
  httpd_register_uri_handler(web,&root);httpd_register_uri_handler(web,&capture);
  conf.server_port=81;conf.ctrl_port+=1;
  if(httpd_start(&stream,&conf)!=ESP_OK)return false;
  httpd_uri_t uri={};uri.uri="/stream";uri.method=HTTP_GET;uri.handler=streamHandler;
  return httpd_register_uri_handler(stream,&uri)==ESP_OK;
}
void setup(){
  Serial.begin(115200);Serial.setDebugOutput(false);delay(500);
  Serial.println("ESP32-S3 PREVIEW | motors not controlled | I2C lane data disabled");
  cameraLock=xSemaphoreCreateMutex();
  if(!cameraLock || !psramFound()){Serial.println("ERROR PSRAM/mutex unavailable");return;}
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
  WiFi.mode(WIFI_AP);WiFi.setSleep(false);
  WiFi.softAPConfig(IPAddress(192,168,5,1),IPAddress(192,168,5,1),IPAddress(255,255,255,0));
  if(!WiFi.softAP("HW_ESP32S3CAM")){Serial.println("ERROR WiFi AP");return;}
  if(!startServers()){Serial.println("ERROR HTTP server");return;}
  Serial.println("PREVIEW READY | WiFi=HW_ESP32S3CAM | http://192.168.5.1/ | stream port 81");
  Serial.println("USB commands: J=snapshot JPEG, S=status");
}
void loop(){
  if(Serial.available()){
    char cmd=Serial.read();
    if(cmd=='S')Serial.printf("STATUS camera=%u clients=%u frames=%u\n",cameraOK,WiFi.softAPgetStationNum(),unsigned(frames));
    if(cmd=='J'){
      uint8_t *jpg=nullptr;size_t size=0;
      if(getJPEG(&jpg,&size)){
        Serial.printf("JPEG %u\n",unsigned(size));Serial.write(jpg,size);Serial.flush();free(jpg);
      }else Serial.println("ERROR snapshot");
    }
  }
  delay(5);
}
