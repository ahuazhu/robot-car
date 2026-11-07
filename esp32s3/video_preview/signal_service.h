#pragma once
#include "start_signal.h"
static SemaphoreHandle_t signalLock, jpegLock;
static start_signal::Detector signalDetector;
static String signalKey,signalSession,lastSignalSession;
static char signalBoot[17];
static uint32_t signalLease=0,signalFrameAt=0;
static int64_t lastCameraTimestamp=-1;
static uint32_t signalProcessUs=0,signalMaxProcessUs=0;
static uint8_t *latestJPEG=nullptr;
static size_t latestJPEGSize=0;
static uint32_t latestJPEGAt=0;
static bool hexString(const char *s,size_t n){if(!s||strlen(s)!=n)return false;for(size_t i=0;i<n;++i)if(!isxdigit((unsigned char)s[i]))return false;return true;}
static void cancelSignal(){signalDetector.cancel();lastSignalSession=signalSession;signalSession="";}
static void pairSignal(const char *data){
  cJSON *root=cJSON_Parse(data);const cJSON *key=root?cJSON_GetObjectItemCaseSensitive(root,"key"):nullptr;
  if(!cJSON_IsString(key)||!hexString(key->valuestring,64)){if(root)cJSON_Delete(root);Serial.println("PAIR ERROR invalid_key");return;}
  xSemaphoreTake(signalLock,portMAX_DELAY);
  Preferences saved;bool ok=false;
  if(saved.begin("start_signal",false)){saved.putString("key",key->valuestring);ok=saved.getString("key","")==key->valuestring;saved.end();}
  if(ok){signalKey=key->valuestring;cancelSignal();}
  xSemaphoreGive(signalLock);cJSON_Delete(root);Serial.println(ok?"PAIR OK":"PAIR ERROR storage_failed");
}
static String signalJSON(){
  xSemaphoreTake(signalLock,portMAX_DELAY);
  uint32_t age=signalDetector.sequence?uint32_t(millis()-signalFrameAt):UINT32_MAX;
  String data="{\"boot\":\""+String(signalBoot)+"\",\"paired\":"+(signalKey.length()==64?"true":"false")+",\"session\":\""+signalSession+"\",\"armed\":"+(signalDetector.armed?"true":"false")+",\"confirmed\":"+(signalDetector.confirmed?"true":"false");
  data+=",\"sequence\":"+String(signalDetector.sequence)+",\"frame_age_ms\":"+String(age)+",\"confirmation_frames\":"+String(signalDetector.hits)+",\"confirmed_ms\":"+String(signalDetector.elapsed);
  data+=",\"phase\":\""+String(signalDetector.phaseName())+"\",\"direction\":\"DARKER\",\"trigger_direction\":\"DARKER\",\"protocol\":2,\"control_port\":81,\"roi\":["+String(start_signal::Detector::X)+","+String(start_signal::Detector::Y)+","+String(start_signal::Detector::RW)+","+String(start_signal::Detector::RH)+"],\"required_frames\":3,\"required_ms\":400";
  data+=",\"target_locked\":"+String(signalDetector.baseline?"true":"false")+",\"target_roi\":["+String(start_signal::Detector::X+signalDetector.target.x*2)+","+String(start_signal::Detector::Y+signalDetector.target.y*2)+","+String(signalDetector.target.w*2)+","+String(signalDetector.target.h*2)+"]";
  data+=",\"contrast\":"+String(signalDetector.contrast,1)+",\"baseline_contrast\":"+String(signalDetector.baselineContrast,1)+",\"noise\":"+String(signalDetector.noise,1);
  data+=",\"rejection_reason\":\""+String(signalDetector.rejectionReason)+"\",\"background_change\":"+String(signalDetector.backgroundChangeL,1)+",\"background_residual\":"+String(signalDetector.backgroundResidualL,1)+",\"darkened_fraction\":"+String(signalDetector.darkenedFraction,3)+",\"required_drop\":"+String(signalDetector.requiredDrop,1);
  data+=",\"segmentation\":\"LUMA_CHROMA_COMPONENT\",\"mask_pixels\":"+String(signalDetector.baseline?signalDetector.maskPixels:0)+",\"exposure_gain\":"+String(signalDetector.exposureGain,3);
  data+=",\"processing_us\":"+String(signalProcessUs)+",\"max_processing_us\":"+String(signalMaxProcessUs);
  data+=",\"event_age_ms\":"+String(signalDetector.eventAge(millis()))+",\"maximum_event_age_ms\":0,\"holds_confirmed_until_cancel\":true,\"preview_url\":\""+(WiFi.status()==WL_CONNECTED?String("http://")+WiFi.localIP().toString()+"/":String(""))+"\"}";
  xSemaphoreGive(signalLock);return data;
}
static bool applySignalCommand(const char *op,const char *session,const char *boot){
  bool valid=op&&hexString(session,32)&&boot&&strcmp(boot,signalBoot)==0;
  bool accepted=false;
  if(valid){
    xSemaphoreTake(signalLock,portMAX_DELAY);
    if(strcmp(op,"arm")==0){
      if(signalSession==session && signalDetector.armed)accepted=true;
      else if(!signalDetector.armed && lastSignalSession!=session){signalSession=session;signalDetector.arm();accepted=true;}
    }else if(strcmp(op,"poll")==0)accepted=signalSession==session&&signalDetector.armed;
    else if(strcmp(op,"cancel")==0){
      if(signalSession==session){cancelSignal();accepted=true;}
      else if(signalSession.length()==0){lastSignalSession=session;accepted=true;}
    }
    if(accepted)signalLease=millis();
    xSemaphoreGive(signalLock);
  }
  return accepted;
}
static esp_err_t signalControl(httpd_req_t *r){
  char key[65]={};
  if(httpd_req_get_hdr_value_len(r,"X-Signal-Key")!=64 || httpd_req_get_hdr_value_str(r,"X-Signal-Key",key,sizeof(key))!=ESP_OK)return httpd_resp_send_err(r,HTTPD_403_FORBIDDEN,"Pairing required");
  xSemaphoreTake(signalLock,portMAX_DELAY);bool authorized=signalKey.length()==64;unsigned diff=0;
  if(authorized)for(int i=0;i<64;++i)diff|=key[i]^signalKey[i];
  xSemaphoreGive(signalLock);memset(key,0,sizeof(key));
  if(!authorized||diff)return httpd_resp_send_err(r,HTTPD_403_FORBIDDEN,"Pairing required");
  if(r->content_len<=0||r->content_len>192)return httpd_resp_send_err(r,HTTPD_400_BAD_REQUEST,"Invalid command");
  char body[193]={};int used=0;
  while(used<r->content_len){int n=httpd_req_recv(r,body+used,r->content_len-used);if(n<=0)return ESP_FAIL;used+=n;}
  cJSON *root=cJSON_Parse(body);const cJSON *op=root?cJSON_GetObjectItemCaseSensitive(root,"op"):nullptr,*session=root?cJSON_GetObjectItemCaseSensitive(root,"session"):nullptr,*boot=root?cJSON_GetObjectItemCaseSensitive(root,"boot"):nullptr;
  bool valid=cJSON_IsString(op)&&cJSON_IsString(session)&&hexString(session->valuestring,32)&&cJSON_IsString(boot)&&strcmp(boot->valuestring,signalBoot)==0;
  bool accepted=valid && applySignalCommand(op->valuestring,session->valuestring,boot->valuestring);
  if(root)cJSON_Delete(root);
  if(!accepted){httpd_resp_set_status(r,"409 Conflict");return httpd_resp_sendstr(r,"Session mismatch; rearm required");}
  String data=signalJSON();httpd_resp_set_type(r,"application/json");httpd_resp_set_hdr(r,"Cache-Control","no-store");return httpd_resp_send(r,data.c_str(),data.length());
}
static void cameraProducer(void*){
  cameraWatchdogReady.store(esp_task_wdt_status(nullptr)==ESP_OK||esp_task_wdt_add(nullptr)==ESP_OK);
  TickType_t due=xTaskGetTickCount();
  for(;;){
    if(cameraWatchdogReady.load())esp_task_wdt_reset();
    camera_fb_t *fb=esp_camera_fb_get();
    if(fb){
      xSemaphoreTake(signalLock,portMAX_DELAY);
      if(signalDetector.armed&&!signalDetector.confirmed&&uint32_t(millis()-signalLease)>2000)cancelSignal();
      int64_t stamp=int64_t(fb->timestamp.tv_sec)*1000000+fb->timestamp.tv_usec;
      int64_t age=esp_timer_get_time()-stamp;
      if(fb->format==PIXFORMAT_RGB565 && fb->width==320 && fb->height==240 && fb->len>=320*240*2 && stamp>lastCameraTimestamp && age>=0 && age<=400000){int64_t began=esp_timer_get_time();signalDetector.process(fb->buf,millis());signalProcessUs=uint32_t(esp_timer_get_time()-began);if(signalProcessUs>signalMaxProcessUs)signalMaxProcessUs=signalProcessUs;signalFrameAt=millis()-uint32_t(age/1000);lastCameraTimestamp=stamp;}
      xSemaphoreGive(signalLock);
      uint8_t *jpg=nullptr;size_t size=0;bool ok=frame2jpg(fb,70,&jpg,&size);esp_camera_fb_return(fb);
      if(ok){xSemaphoreTake(jpegLock,portMAX_DELAY);free(latestJPEG);latestJPEG=jpg;latestJPEGSize=size;latestJPEGAt=millis();++frames;xSemaphoreGive(jpegLock);}
    }
    vTaskDelayUntil(&due,pdMS_TO_TICKS(100));
    // If a slow frame consumed the period, don't run a catch-up burst.
    if(int32_t(xTaskGetTickCount()-due)>100)due=xTaskGetTickCount();
  }
}
static void initSignal(){
  signalLock=xSemaphoreCreateMutex();jpegLock=xSemaphoreCreateMutex();
  snprintf(signalBoot,sizeof(signalBoot),"%08lx%08lx",(unsigned long)esp_random(),(unsigned long)esp_random());
  Preferences saved;if(saved.begin("start_signal",true)){signalKey=saved.getString("key","");saved.end();}
}
