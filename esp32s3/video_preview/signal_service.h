#pragma once
#include "start_signal.h"
static SemaphoreHandle_t signalLock, jpegLock;
static start_signal::Detector signalDetector;
static String signalKey,signalSession,lastSignalSession;
static bool signalInitialLampOn=false;
static const char *signalInitialLampState(){return signalInitialLampOn?"ON":"OFF";}
static char signalBoot[17];
// Caller holds signalLock; never emit the key in status or errors.
static bool signalKeyMatches(const char *key){
  if(!key||strlen(key)!=64||signalKey.length()!=64)return false;
  unsigned difference=0;for(int i=0;i<64;++i)difference|=key[i]^signalKey[i];
  return difference==0;
}
#ifdef ROBOT_LAMPNET_FACE
static start_signal::Detector *modelTestDetector=nullptr;
static bool modelTestEnabled=false,manualLampEnabled=false;
static int manualLampROI[4]={100,40,100,80};
static bool validManualROI(const int *r){return r[0]>=start_signal::Detector::X&&r[1]>=start_signal::Detector::Y&&r[0]<=start_signal::Detector::X+start_signal::Detector::RW&&r[1]<=start_signal::Detector::Y+start_signal::Detector::RH&&r[2]>=16&&r[3]>=16&&r[2]<=start_signal::Detector::RW&&r[3]<=start_signal::Detector::RH&&r[0]+r[2]<=start_signal::Detector::X+start_signal::Detector::RW&&r[1]+r[3]<=start_signal::Detector::Y+start_signal::Detector::RH;}
static String lampConfiguration(){return String("{\"manual\":")+(manualLampEnabled?"true":"false")+",\"roi\":["+String(manualLampROI[0])+","+String(manualLampROI[1])+","+String(manualLampROI[2])+","+String(manualLampROI[3])+"]}";}
enum class LampConfigurationResult { OK,INVALID,ARMED,STORAGE,BOOT_CHANGED,UNAUTHORIZED };
struct LampConfigurationUpdate {bool manual=false;int roi[4]={};};
static bool parseLampConfiguration(const cJSON *root,LampConfigurationUpdate &update){
  const cJSON *mode=cJSON_GetObjectItemCaseSensitive(root,"manual"),*box=cJSON_GetObjectItemCaseSensitive(root,"roi");
  if(!cJSON_IsObject(root)||!cJSON_IsBool(mode))return false;
  update.manual=cJSON_IsTrue(mode);
  if(!update.manual)return true; // Retain the last saved manual rectangle.
  if(!cJSON_IsArray(box)||cJSON_GetArraySize(box)!=4)return false;
  for(int i=0;i<4;++i){const cJSON *value=cJSON_GetArrayItem(box,i);
    if(!cJSON_IsNumber(value)||!isfinite(value->valuedouble)||value->valuedouble!=value->valueint)return false;
    update.roi[i]=value->valueint;
  }
  return validManualROI(update.roi);
}
static LampConfigurationResult saveLampConfiguration(const LampConfigurationUpdate &update,const char *boot=nullptr,const char *key=nullptr){
  xSemaphoreTake(signalLock,portMAX_DELAY);
  LampConfigurationResult result=LampConfigurationResult::OK;
  // Recheck authentication, boot and ARM under the same lock as persistence.
  if(key&&!signalKeyMatches(key))result=LampConfigurationResult::UNAUTHORIZED;
  else if(boot&&strcmp(boot,signalBoot)!=0)result=LampConfigurationResult::BOOT_CHANGED;
  else if(signalDetector.armed)result=LampConfigurationResult::ARMED;
  else{
    Preferences saved;bool stored=saved.begin("lamp_roi",false);
    if(stored){int value[5]={int(update.manual),manualLampROI[0],manualLampROI[1],manualLampROI[2],manualLampROI[3]},check[5]={};
      if(update.manual)memcpy(value+1,update.roi,sizeof(update.roi));
      stored=saved.putBytes("config",value,sizeof(value))==sizeof(value)&&saved.getBytes("config",check,sizeof(check))==sizeof(check)&&memcmp(value,check,sizeof(value))==0;
      saved.end();
    }
    if(stored){manualLampEnabled=update.manual;if(update.manual)memcpy(manualLampROI,update.roi,sizeof(manualLampROI));if(modelTestDetector)modelTestDetector->cancel();modelTestEnabled=false;}
    else result=LampConfigurationResult::STORAGE;
  }
  xSemaphoreGive(signalLock);return result;
}
static const char *lampConfigurationError(LampConfigurationResult result){
  switch(result){
    case LampConfigurationResult::INVALID:return "invalid_configuration";
    case LampConfigurationResult::ARMED:return "native_armed_stop_first";
    case LampConfigurationResult::STORAGE:return "storage_failed";
    case LampConfigurationResult::BOOT_CHANGED:return "boot_changed";
    case LampConfigurationResult::UNAUTHORIZED:return "pairing_required";
    default:return "";
  }
}
static void configureLampROI(const char *data){
  cJSON *root=cJSON_Parse(data);LampConfigurationUpdate update;
  bool valid=parseLampConfiguration(root,update);if(root)cJSON_Delete(root);
  if(!valid){Serial.println("ROI ERROR invalid_rectangle");return;}
  LampConfigurationResult result=saveLampConfiguration(update);
  if(result!=LampConfigurationResult::OK){Serial.println(String("ROI ERROR ")+lampConfigurationError(result));return;}
  xSemaphoreTake(signalLock,portMAX_DELAY);String state=lampConfiguration();xSemaphoreGive(signalLock);
  Serial.println("ROI OK "+state);
}
static void applyModelHint(start_signal::Detector &detector,uint32_t now){
  detector.setManualROI(manualLampROI[0],manualLampROI[1],manualLampROI[2],manualLampROI[3],manualLampEnabled);
  if(manualLampEnabled){detector.requireModelLocator(false);return;}
  auto s=lampnet::snapshot();detector.requireModelLocator(true);
  detector.setModelHint(s.x,s.y,s.w,s.h,s.valid&&s.sourceAt&&uint32_t(now-s.sourceAt)<=2500);
}
// Drawing hint only: read existing model output, never request an inference.
static String lampCandidateHint(uint32_t now){
  auto source=lampnet::snapshot();uint32_t age=source.sourceAt?uint32_t(now-source.sourceAt):UINT32_MAX;
  int roi[4]={source.x,source.y,source.w,source.h};
  bool valid=source.selfTestOK&&source.valid&&source.sourceAt&&age<=2500&&validManualROI(roi);
  if(!valid)memset(roi,0,sizeof(roi));
  return String("{\"valid\":")+(valid?"true":"false")+",\"roi\":["+String(roi[0])+","+String(roi[1])+","+String(roi[2])+","+String(roi[3])+"],\"age_ms\":"+String(age)+"}";
}
static String modelTestJSON(){
  xSemaphoreTake(signalLock,portMAX_DELAY);String out="{\"configuration\":"+lampConfiguration()+",\"enabled\":"+String(modelTestEnabled?"true":"false")+",\"controls_start\":false";
  out+=",\"native_armed\":"+String(signalDetector.armed?"true":"false")+",\"native_confirmed\":"+String(signalDetector.confirmed?"true":"false");
  if(modelTestDetector){auto &d=*modelTestDetector;out+=",\"phase\":\""+String(d.phaseName())+"\",\"locked\":"+String(d.baseline?"true":"false")+",\"confirmed\":"+String(d.confirmed?"true":"false")+",\"hits\":"+String(d.hits)+",\"elapsed_ms\":"+String(d.elapsed)+",\"sequence\":"+String(d.sequence)+",\"reason\":\""+String(d.rejectionReason)+"\",\"roi\":["+String(start_signal::Detector::X+d.target.x*start_signal::Detector::STEP)+","+String(start_signal::Detector::Y+d.target.y*start_signal::Detector::STEP)+","+String(d.target.w*start_signal::Detector::STEP)+","+String(d.target.h*start_signal::Detector::STEP)+"],\"drop_threshold\":"+String(d.requiredDrop)+",\"dark_fraction\":"+String(d.darkenedFraction);}
  if(modelTestDetector){out+=",\"cycle\":\"OFF_ON_OFF\",\"off_baseline_ready\":"+String(modelTestDetector->offBaselineReady?"true":"false")+",\"on_rise\":"+String(modelTestDetector->onRise,1)+",\"brightened_fraction\":"+String(modelTestDetector->brightenedFraction,3);out+=",\"gray_drop\":"+String(modelTestDetector->grayDrop,1)+",\"value_drop\":"+String(modelTestDetector->valueDrop,1)+",\"saturation_change\":"+String(modelTestDetector->saturationChange,3);}
  out+="}";xSemaphoreGive(signalLock);return out;
}
static void modelTestCommand(bool enable){
  xSemaphoreTake(signalLock,portMAX_DELAY);
  if(!signalDetector.armed){
    if(enable&&!modelTestDetector){void *p=heap_caps_malloc(sizeof(start_signal::Detector),MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);if(p)modelTestDetector=new(p) start_signal::Detector();}
    modelTestEnabled=enable&&modelTestDetector;
    if(modelTestDetector){if(modelTestEnabled){modelTestDetector->requireModelLocator(true);modelTestDetector->requireOffOnOff(true);modelTestDetector->arm();}else modelTestDetector->cancel();}
  }
  xSemaphoreGive(signalLock);Serial.println("LAMP_TEST "+modelTestJSON());
}
#endif
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
  data+=",\"initial_lamp_state\":\""+String(signalInitialLampState())+"\"";
  data+=",\"phase\":\""+String(signalDetector.phaseName())+"\",\"direction\":\"DARKER\",\"trigger_direction\":\"DARKER\",\"protocol\":2,\"control_port\":81,\"roi\":["+String(start_signal::Detector::X)+","+String(start_signal::Detector::Y)+","+String(start_signal::Detector::RW)+","+String(start_signal::Detector::RH)+"],\"required_frames\":3,\"required_ms\":400";
  data+=",\"target_locked\":"+String(signalDetector.baseline?"true":"false")+",\"target_roi\":["+String(start_signal::Detector::X+signalDetector.target.x*start_signal::Detector::STEP)+","+String(start_signal::Detector::Y+signalDetector.target.y*start_signal::Detector::STEP)+","+String(signalDetector.target.w*start_signal::Detector::STEP)+","+String(signalDetector.target.h*start_signal::Detector::STEP)+"]";
  data+=",\"contrast\":"+String(signalDetector.contrast,1)+",\"baseline_contrast\":"+String(signalDetector.baselineContrast,1)+",\"noise\":"+String(signalDetector.noise,1);
  data+=",\"rejection_reason\":\""+String(signalDetector.rejectionReason)+"\",\"background_change\":"+String(signalDetector.backgroundChangeL,1)+",\"background_residual\":"+String(signalDetector.backgroundResidualL,1)+",\"darkened_fraction\":"+String(signalDetector.darkenedFraction,3)+",\"required_drop\":"+String(signalDetector.requiredDrop,1);
  data+=",\"segmentation\":\""+String(
#ifdef ROBOT_LAMPNET_FACE
    "MODEL_FACE_LUMA_CHROMA"
#else
    "LUMA_CHROMA_COMPONENT"
#endif
  )+"\",\"mask_pixels\":"+String(signalDetector.baseline?signalDetector.maskPixels:0)+",\"exposure_gain\":"+String(signalDetector.exposureGain,3);
  data+=",\"processing_us\":"+String(signalProcessUs)+",\"max_processing_us\":"+String(signalMaxProcessUs);
#ifdef ROBOT_LAMPNET_FACE
  data+=",\"lamp_configuration\":"+lampConfiguration();
  data+=",\"candidate_hint\":"+lampCandidateHint(millis());
#endif
  data+=",\"event_age_ms\":"+String(signalDetector.eventAge(millis()))+",\"maximum_event_age_ms\":0,\"holds_confirmed_until_cancel\":true,\"preview_url\":\""+(WiFi.status()==WL_CONNECTED?String("http://")+WiFi.localIP().toString()+"/":String(""))+"\"}";
  xSemaphoreGive(signalLock);return data;
}
static bool applySignalCommand(const char *op,const char *session,const char *boot,const char *initialLampState="OFF"){
  bool validInitial=initialLampState&&(strcmp(initialLampState,"OFF")==0||strcmp(initialLampState,"ON")==0);
  bool valid=op&&hexString(session,32)&&boot&&strcmp(boot,signalBoot)==0;
  bool accepted=false;
  if(valid){
    xSemaphoreTake(signalLock,portMAX_DELAY);
    if(strcmp(op,"arm")==0&&validInitial){
#ifdef ROBOT_LAMPNET_FACE
      if(modelTestEnabled){modelTestEnabled=false;if(modelTestDetector)modelTestDetector->cancel();}
#endif
      bool initialOn=strcmp(initialLampState,"ON")==0;
      if(signalSession==session && signalDetector.armed)accepted=initialOn==signalInitialLampOn;
      else if(!signalDetector.armed && lastSignalSession!=session){signalSession=session;
      signalInitialLampOn=initialOn;signalDetector.requireOffOnOff(!initialOn);
      signalDetector.arm();accepted=true;}
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
static esp_err_t signalError(httpd_req_t *request,const char *status,const char *error){
  httpd_resp_set_status(request,status);httpd_resp_set_type(request,"application/json");httpd_resp_set_hdr(request,"Cache-Control","no-store");
  String body=String("{\"error\":\"")+error+"\"}";return httpd_resp_send(request,body.c_str(),body.length());
}
static esp_err_t signalControl(httpd_req_t *r){
  char key[65]={};
  if(httpd_req_get_hdr_value_len(r,"X-Signal-Key")!=64 || httpd_req_get_hdr_value_str(r,"X-Signal-Key",key,sizeof(key))!=ESP_OK)return signalError(r,"403 Forbidden","pairing_required");
  xSemaphoreTake(signalLock,portMAX_DELAY);bool authorized=signalKeyMatches(key);xSemaphoreGive(signalLock);
  if(!authorized){memset(key,0,sizeof(key));return signalError(r,"403 Forbidden","pairing_required");}
  if(r->content_len<=0||r->content_len>192){memset(key,0,sizeof(key));return signalError(r,"400 Bad Request","invalid_request");}
  char body[193]={};int used=0;
  while(used<r->content_len){int n=httpd_req_recv(r,body+used,r->content_len-used);if(n<=0){memset(key,0,sizeof(key));return ESP_FAIL;}used+=n;}
  cJSON *root=cJSON_Parse(body);const cJSON *op=root?cJSON_GetObjectItemCaseSensitive(root,"op"):nullptr,*session=root?cJSON_GetObjectItemCaseSensitive(root,"session"):nullptr,*boot=root?cJSON_GetObjectItemCaseSensitive(root,"boot"):nullptr;
#ifdef ROBOT_LAMPNET_FACE
  if(cJSON_IsString(op)&&strcmp(op->valuestring,"configure")==0){
    LampConfigurationUpdate update;LampConfigurationResult result=LampConfigurationResult::INVALID;
    bool envelope=cJSON_IsObject(root)&&cJSON_IsString(boot);
    if(envelope&&parseLampConfiguration(cJSON_GetObjectItemCaseSensitive(root,"configuration"),update))result=saveLampConfiguration(update,boot->valuestring,key);
    memset(key,0,sizeof(key));if(root)cJSON_Delete(root);
    if(!envelope)return signalError(r,"400 Bad Request","invalid_request");
    if(result!=LampConfigurationResult::OK){
      const char *status=result==LampConfigurationResult::ARMED||result==LampConfigurationResult::BOOT_CHANGED?"409 Conflict":result==LampConfigurationResult::UNAUTHORIZED?"403 Forbidden":result==LampConfigurationResult::STORAGE?"500 Internal Server Error":"400 Bad Request";
      return signalError(r,status,lampConfigurationError(result));
    }
    String data=signalJSON();httpd_resp_set_type(r,"application/json");httpd_resp_set_hdr(r,"Cache-Control","no-store");return httpd_resp_send(r,data.c_str(),data.length());
  }
#endif
  memset(key,0,sizeof(key));
  const cJSON *initial=root?cJSON_GetObjectItemCaseSensitive(root,"initial_lamp_state"):nullptr;
  const char *initialState=initial&&cJSON_IsString(initial)?initial->valuestring:initial?nullptr:"OFF";
  if(cJSON_IsString(op)&&strcmp(op->valuestring,"arm")==0&&(!initialState||(strcmp(initialState,"OFF")!=0&&strcmp(initialState,"ON")!=0))){
    if(root)cJSON_Delete(root);return signalError(r,"400 Bad Request","invalid_initial_lamp_state");
  }
  bool valid=cJSON_IsString(op)&&cJSON_IsString(session)&&hexString(session->valuestring,32)&&cJSON_IsString(boot)&&strcmp(boot->valuestring,signalBoot)==0;
  bool accepted=valid && applySignalCommand(op->valuestring,session->valuestring,boot->valuestring,initialState);
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
      if(fb->format==PIXFORMAT_RGB565 && fb->width==320 && fb->height==240 && fb->len>=320*240*2 && stamp>lastCameraTimestamp && age>=0 && age<=400000){
#ifdef ROBOT_LAMPNET_FACE
        applyModelHint(signalDetector,millis());
        if(modelTestEnabled&&modelTestDetector){applyModelHint(*modelTestDetector,millis());modelTestDetector->process(fb->buf,millis());}
#endif
        int64_t began=esp_timer_get_time();signalDetector.process(fb->buf,millis());signalProcessUs=uint32_t(esp_timer_get_time()-began);if(signalProcessUs>signalMaxProcessUs)signalMaxProcessUs=signalProcessUs;signalFrameAt=millis()-uint32_t(age/1000);lastCameraTimestamp=stamp;}
      bool needModel=!signalDetector.armed||!signalDetector.baseline;
#ifdef ROBOT_LAMPNET_FACE
      if(modelTestEnabled&&modelTestDetector&&!signalDetector.armed)needModel=!modelTestDetector->baseline;
#endif
#ifdef ROBOT_LAMPNET_FACE
      if(manualLampEnabled)needModel=false;
#endif
      xSemaphoreGive(signalLock);
#ifdef ROBOT_LAMPNET_SHADOW
      if(needModel&&fb->format==PIXFORMAT_RGB565 && fb->width==320 && fb->height==240 && fb->len>=320*240*2 && age>=0 && age<=400000)lampnet::offer(fb->buf,millis()-uint32_t(age/1000));
#endif
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
#ifdef ROBOT_LAMPNET_FACE
  Preferences lampSaved;if(lampSaved.begin("lamp_roi",true)){int value[5]={};bool ok=lampSaved.getBytesLength("config")==sizeof(value)&&lampSaved.getBytes("config",value,sizeof(value))==sizeof(value);lampSaved.end();if(ok&&validManualROI(value+1)){manualLampEnabled=value[0]==1;memcpy(manualLampROI,value+1,sizeof(manualLampROI));}}
#endif
  snprintf(signalBoot,sizeof(signalBoot),"%08lx%08lx",(unsigned long)esp_random(),(unsigned long)esp_random());
  Preferences saved;if(saved.begin("start_signal",true)){signalKey=saved.getString("key","");saved.end();}
}
