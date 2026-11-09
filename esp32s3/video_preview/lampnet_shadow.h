#pragma once
// On-device localization. Signal service owns the temporal start decision.
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_heap_caps.h"
#include "lampnet_model_data.h"
#include "lamp_face_locator.h"
#include <math.h>
#include <algorithm>
#include <new>
#ifdef ROBOT_LAMPNET_ESP_NN
#include "lampnet_esp_nn.h"
#endif

namespace lampnet {
static constexpr int MODEL_SIDE=160, GRID=40;
#ifdef ROBOT_LAMPNET_FULL_FRAME
static constexpr int VIEW_X=0,VIEW_Y=0,VIEW_W=320,VIEW_H=240;
#else
static constexpr int VIEW_X=80,VIEW_Y=0,VIEW_W=160,VIEW_H=128;
#endif
static constexpr size_t ARENA_BYTES=1024*1024;
static tflite::MicroInterpreter *interpreter=nullptr;
static TfLiteTensor *input=nullptr,*output=nullptr;
static uint8_t *arena=nullptr;
static TaskHandle_t worker=nullptr;
static std::atomic<bool> ready(false),busy(true);
static int8_t lut[256];
static uint8_t reverseRGB[256],*rgbGrid=nullptr,*probabilityGrid=nullptr;
static lamp_face::Locator *faceLocator=nullptr;
static uint32_t offeredAt=0;
static portMUX_TYPE statsMux=portMUX_INITIALIZER_UNLOCKED;
struct Statistics {
  bool initialized=false,valid=false,selfTestOK=false;
  uint32_t arenaUsed=0,inferences=0,lastUs=0,maxUs=0,completedAt=0,sourceAt=0;
  uint32_t preprocessingUs=0,maskCells=0,refinementUs=0;
  int coarseX=0,coarseY=0,coarseW=0,coarseH=0;bool coarseValid=false;
  unsigned refinementMethod=0,coreThreshold=141;
  uint32_t convUs=0,depthwiseUs=0,addUs=0,resizeUs=0,otherUs=0,yieldUs=0;
  unsigned goldenCount=0;
  struct Golden {uint32_t invokeUs=0,maximum=0;float mean=0;bool ok=false;} golden[8];
  int x=0,y=0,w=0,h=0;
  float confidence=0;
  const char *error="STARTING";
};
static Statistics stats;

// Time actual operators separately from scheduler yields. Accelerated
// kernels only yield after an 8ms timeslice, rather than sleeping per node.
class YieldProfiler: public tflite::MicroProfilerInterface {
 public:
  uint32_t convUs=0,depthwiseUs=0,addUs=0,resizeUs=0,otherUs=0,yieldUs=0,lastSlice=0;
  unsigned kind=0;
  void reset(){convUs=depthwiseUs=addUs=resizeUs=otherUs=yieldUs=0;lastSlice=uint32_t(esp_timer_get_time());}
  uint32_t BeginEvent(const char *tag) override {
    kind=strcmp(tag,"CONV_2D")==0?1:strcmp(tag,"DEPTHWISE_CONV_2D")==0?2:strcmp(tag,"ADD")==0?3:strcmp(tag,"RESIZE_BILINEAR")==0?4:0;
    return uint32_t(esp_timer_get_time());
  }
  void EndEvent(uint32_t began) override {
    uint32_t now=uint32_t(esp_timer_get_time()),elapsed=now-began;
    if(kind==1)convUs+=elapsed;else if(kind==2)depthwiseUs+=elapsed;else if(kind==3)addUs+=elapsed;else if(kind==4)resizeUs+=elapsed;else otherUs+=elapsed;
#ifdef ROBOT_LAMPNET_ESP_NN
    if(uint32_t(now-lastSlice)<8000)return;
#endif
    vTaskDelay(1);lastSlice=uint32_t(esp_timer_get_time());yieldUs+=lastSlice-now;
  }
};
static YieldProfiler profiler;

static Statistics snapshot(){portENTER_CRITICAL(&statsMux);Statistics s=stats;portEXIT_CRITICAL(&statsMux);return s;}
static void failure(const char *message){portENTER_CRITICAL(&statsMux);stats.error=message;stats.valid=false;portEXIT_CRITICAL(&statsMux);ready.store(false);}

static String statusJSON(){
  Statistics s=snapshot();String out="{\"mode\":\""+String(
#ifdef ROBOT_LAMPNET_FACE
    "MODEL_LOCATOR"
#else
    "SHADOW_ONLY"
#endif
  )+"\",\"controls_start\":"+String(
#ifdef ROBOT_LAMPNET_FACE
    "true"
#else
    "false"
#endif
  )+",\"ready\":"+String(ready.load()?"true":"false");
  out+=",\"busy\":"+String(busy.load()?"true":"false")+",\"model\":\"LampNet160-INT8\",\"model_bytes\":"+String(kLampModelLen);
#ifdef ROBOT_LAMPNET_ESP_NN
  out+=",\"engine\":\"ESP_NN_S3_SIMD\",\"accelerated_conv_nodes\":"+String(lampnet_accel::convNodes)+",\"accelerated_depthwise_nodes\":"+String(lampnet_accel::depthwiseNodes)+",\"accelerated_add_nodes\":"+String(lampnet_accel::addNodes)+",\"dual_core\":"+String(esp_nn_dual_core_active()?"true":"false")+",\"nn_scratch_bytes\":"+String(lampnet_accel::scratchBytes)+",\"nn_internal_scratch_bytes\":"+String(lampnet_accel::internalBytes);
#else
  out+=",\"engine\":\"PORTABLE\"";
#endif
  out+=",\"arena_reserved\":"+String(ARENA_BYTES)+",\"arena_used\":"+String(s.arenaUsed)+",\"selftest_ok\":"+String(s.selfTestOK?"true":"false");
  out+=",\"inferences\":"+String(s.inferences)+",\"inference_us\":"+String(s.lastUs)+",\"max_inference_us\":"+String(s.maxUs)+",\"preprocess_us\":"+String(s.preprocessingUs);
  out+=",\"operator_us\":{\"conv\":"+String(s.convUs)+",\"depthwise\":"+String(s.depthwiseUs)+",\"add\":"+String(s.addUs)+",\"resize\":"+String(s.resizeUs)+",\"other\":"+String(s.otherUs)+",\"scheduler_yield\":"+String(s.yieldUs)+"},\"golden\":[";
  for(unsigned i=0;i<s.goldenCount;++i){if(i)out+=",";out+="{\"name\":\""+String(kLampGolden[i].name)+"\",\"invoke_us\":"+String(s.golden[i].invokeUs)+",\"mean_abs_quant_error\":"+String(s.golden[i].mean,3)+",\"max_abs_quant_error\":"+String(s.golden[i].maximum)+",\"numeric_match\":"+String(s.golden[i].ok?"true":"false")+"}";}
  out+="]";
  out+=",\"locator\":\"DUAL_THRESHOLD_COLOUR_CORE\",\"refinement_us\":"+String(s.refinementUs)+",\"refinement_method\":\""+String(lamp_face::methodName(lamp_face::Method(s.refinementMethod)))+"\",\"core_threshold\":"+String(s.coreThreshold/256.f,3)+",\"coarse_valid\":"+String(s.coarseValid?"true":"false")+",\"coarse_roi\":["+String(s.coarseX)+","+String(s.coarseY)+","+String(s.coarseW)+","+String(s.coarseH)+"]";
  out+=",\"search_roi\":["+String(VIEW_X)+","+String(VIEW_Y)+","+String(VIEW_W)+","+String(VIEW_H)+"]";
  out+=",\"candidate_valid\":"+String(s.valid?"true":"false")+",\"candidate_roi\":["+String(s.x)+","+String(s.y)+","+String(s.w)+","+String(s.h)+"],\"confidence\":"+String(s.confidence,3)+",\"mask_cells\":"+String(s.maskCells);
  out+=",\"source_age_ms\":"+String(s.sourceAt?uint32_t(millis()-s.sourceAt):UINT32_MAX)+",\"error\":\""+String(s.error)+"\",\"free_heap\":"+String(ESP.getFreeHeap())+",\"free_psram\":"+String(ESP.getFreePsram())+"}";return out;
}

static void inspect(uint32_t elapsed,uint32_t captured){
  int64_t began=esp_timer_get_time();
  for(int i=0;i<GRID*GRID;++i)probabilityGrid[i]=uint8_t(int(output->data.int8[i])+128);
  auto face=faceLocator->locate(probabilityGrid,rgbGrid);uint32_t refinementUs=uint32_t(esp_timer_get_time()-began);
  portENTER_CRITICAL(&statsMux);
  stats.convUs=profiler.convUs;stats.depthwiseUs=profiler.depthwiseUs;stats.addUs=profiler.addUs;stats.resizeUs=profiler.resizeUs;stats.otherUs=profiler.otherUs;stats.yieldUs=profiler.yieldUs;
  stats.inferences++;stats.lastUs=elapsed;stats.maxUs=std::max(stats.maxUs,elapsed);stats.completedAt=millis();stats.sourceAt=captured;
  stats.maskCells=face.area;stats.confidence=face.confidence;stats.valid=face.valid;stats.refinementUs=refinementUs;
  stats.coarseValid=face.coarseValid;stats.coarseX=face.coarseBox.x;stats.coarseY=face.coarseBox.y;stats.coarseW=face.coarseBox.w;stats.coarseH=face.coarseBox.h;stats.refinementMethod=face.method;stats.coreThreshold=face.threshold;
  stats.x=face.box.x;stats.y=face.box.y;stats.w=face.box.w;stats.h=face.box.h;
  stats.error="";portEXIT_CRITICAL(&statsMux);
}

static void run(void*){
  if(kLampGoldenCount>8){failure("TOO_MANY_GOLDEN_CASES");busy.store(false);vTaskDelete(nullptr);return;}
#if defined(ROBOT_LAMPNET_ESP_NN) && defined(ROBOT_LAMPNET_DUAL_CORE)
  // This worker runs on core 0; the optional SIMD worker is pinned on core 1.
  esp_nn_dual_core_enable();
#endif
  bool passed=true;
  for(unsigned i=0;i<kLampGoldenCount;++i){
    memcpy(input->data.int8,kLampGolden[i].input,input->bytes);
    profiler.reset();int64_t began=esp_timer_get_time();TfLiteStatus result=interpreter->Invoke();uint32_t elapsed=uint32_t(esp_timer_get_time()-began);
    if(result!=kTfLiteOk){failure("GOLDEN_INVOKE_FAILED");busy.store(false);vTaskDelete(nullptr);return;}
    unsigned maximum=0;float total=0;
    for(unsigned j=0;j<output->bytes;++j){unsigned delta=abs(int(output->data.int8[j])-int(kLampGolden[i].expected[j]));maximum=std::max(maximum,delta);total+=delta;}
    float mean=total/output->bytes;bool ok=mean<=4&&maximum<=24;passed=passed&&ok;
    portENTER_CRITICAL(&statsMux);stats.golden[i]={elapsed,maximum,mean,ok};stats.goldenCount=i+1;portEXIT_CRITICAL(&statsMux);
    vTaskDelay(pdMS_TO_TICKS(50));
  }
  portENTER_CRITICAL(&statsMux);stats.selfTestOK=passed;if(passed)stats.error="";portEXIT_CRITICAL(&statsMux);
  if(!passed){failure("GOLDEN_NUMERIC_MISMATCH");busy.store(false);vTaskDelete(nullptr);return;}
  busy.store(false);
  for(;;){
    ulTaskNotifyTake(pdTRUE,portMAX_DELAY);
    profiler.reset();int64_t began=esp_timer_get_time();TfLiteStatus result=interpreter->Invoke();uint32_t elapsed=uint32_t(esp_timer_get_time()-began);
    if(result!=kTfLiteOk){failure("LIVE_INVOKE_FAILED");busy.store(false);vTaskDelete(nullptr);return;}
    inspect(elapsed,offeredAt);busy.store(false);
    vTaskDelay(1);
  }
}

static void init(){
  arena=(uint8_t*)heap_caps_malloc(ARENA_BYTES,MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
  if(!arena){failure("PSRAM_ARENA_ALLOCATION_FAILED");return;}
  const tflite::Model *model=tflite::GetModel(kLampModel);
  if(model->version()!=TFLITE_SCHEMA_VERSION){failure("SCHEMA_MISMATCH");return;}
  static tflite::MicroMutableOpResolver<7> resolver;
  #ifdef ROBOT_LAMPNET_ESP_NN
  resolver.AddConv2D(lampnet_accel::conv());resolver.AddDepthwiseConv2D(lampnet_accel::depthwise());resolver.AddAdd(lampnet_accel::add());
#else
  resolver.AddConv2D();resolver.AddDepthwiseConv2D();resolver.AddAdd();
#endif
  resolver.AddConcatenation();resolver.AddPad();resolver.AddResizeBilinear();resolver.AddLogistic();
  interpreter=new(std::nothrow)tflite::MicroInterpreter(model,resolver,arena,ARENA_BYTES,nullptr,&profiler);
  if(!interpreter||interpreter->AllocateTensors()!=kTfLiteOk){failure("TENSOR_ALLOCATION_FAILED");return;}
#ifdef ROBOT_LAMPNET_ESP_NN
  if(!lampnet_accel::allocateWorkspace()){failure("ESP_NN_WORKSPACE_ALLOCATION_FAILED");return;}
#endif
  input=interpreter->input(0);output=interpreter->output(0);
  if(input->type!=kTfLiteInt8||output->type!=kTfLiteInt8||input->bytes!=MODEL_SIDE*MODEL_SIDE*3||output->bytes!=GRID*GRID){failure("TENSOR_LAYOUT_MISMATCH");return;}
  if(output->params.zero_point!=-128||fabsf(output->params.scale-1.f/256)>1e-8f){failure("OUTPUT_QUANTIZATION_MISMATCH");return;}
  for(int value=0;value<256;++value){int q=lroundf((value/127.5f-1)/input->params.scale)+input->params.zero_point;lut[value]=int8_t(std::max(-128,std::min(127,q)));}
  portENTER_CRITICAL(&statsMux);stats.initialized=true;stats.arenaUsed=interpreter->arena_used_bytes();stats.error="SELFTEST_RUNNING";portEXIT_CRITICAL(&statsMux);
  rgbGrid=static_cast<uint8_t*>(heap_caps_malloc(GRID*GRID*3,MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT));
  probabilityGrid=static_cast<uint8_t*>(heap_caps_malloc(GRID*GRID,MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT));
  void *workspace=heap_caps_malloc(sizeof(lamp_face::Locator),MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
  if(!rgbGrid||!probabilityGrid||!workspace){failure("FACE_WORKSPACE_ALLOCATION_FAILED");return;}
  faceLocator=new(workspace) lamp_face::Locator();faceLocator->view(VIEW_X,VIEW_Y,VIEW_W,VIEW_H);
  for(int q=-128;q<128;++q){float value=((q-input->params.zero_point)*input->params.scale+1)*127.5f;reverseRGB[q+128]=uint8_t(std::max(0,std::min(255,int(floorf(value+.5f)))));}
  ready.store(true);
  if(xTaskCreatePinnedToCore(run,"lampnet_probe",16384,nullptr,1,&worker,0)!=pdPASS){failure("WORKER_CREATE_FAILED");return;}
  Serial.println("LAMP_MODEL "+statusJSON());
}

static void offer(const uint8_t *rgb565,uint32_t captured){
  if(!ready.load()||!worker)return;bool expected=false;if(!busy.compare_exchange_strong(expected,true))return;
  int64_t began=esp_timer_get_time();
  for(int y=0;y<MODEL_SIDE;++y){
    float sy=(y+.5f)*VIEW_H/MODEL_SIDE-.5f;int ay=int(floorf(sy));float fy=sy-ay;
    int y0=std::max(0,std::min(VIEW_H-1,ay))+VIEW_Y,y1=std::max(0,std::min(VIEW_H-1,ay+1))+VIEW_Y;
    for(int x=0;x<MODEL_SIDE;++x){
      float sx=(x+.5f)*VIEW_W/MODEL_SIDE-.5f;int ax=int(floorf(sx));float fx=sx-ax;
      int x0=std::max(0,std::min(VIEW_W-1,ax))+VIEW_X,x1=std::max(0,std::min(VIEW_W-1,ax+1))+VIEW_X;
      unsigned at[4]={(y0*320+x0)*2u,(y0*320+x1)*2u,(y1*320+x0)*2u,(y1*320+x1)*2u};
      uint16_t p[4];for(int i=0;i<4;++i)p[i]=(rgb565[at[i]]<<8)|rgb565[at[i]+1];
      const int shift[3]={11,5,0},maximum[3]={31,63,31};
      for(int c=0;c<3;++c){int v[4];for(int i=0;i<4;++i)v[i]=((p[i]>>shift[c])&maximum[c])*255/maximum[c];float top=v[0]+(v[1]-v[0])*fx,bottom=v[2]+(v[3]-v[2])*fx;int value=int(top+(bottom-top)*fy+.5f);input->data.int8[(y*MODEL_SIDE+x)*3+c]=lut[value];}
    }
  }
  // Preserve compact colour evidence before TFLM can reuse input storage.
  for(int y=0;y<GRID;++y)for(int x=0;x<GRID;++x)for(int c=0;c<3;++c){unsigned sum=0;for(int dy=0;dy<4;++dy)for(int dx=0;dx<4;++dx)sum+=reverseRGB[int(input->data.int8[((y*4+dy)*MODEL_SIDE+x*4+dx)*3+c])+128];rgbGrid[(y*GRID+x)*3+c]=(sum+8)/16;}
  offeredAt=captured;uint32_t elapsed=uint32_t(esp_timer_get_time()-began);
  portENTER_CRITICAL(&statsMux);stats.preprocessingUs=elapsed;portEXIT_CRITICAL(&statsMux);
  xTaskNotifyGive(worker);
}
} // namespace lampnet
