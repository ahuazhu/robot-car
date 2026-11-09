#pragma once
// INT8-only TFLM adapters for the unchanged LampNet160 graph.
// Quantization/padding are calculated by TFLM; ESP-NN owns SIMD evaluation.
#include <esp_nn.h>
#include <esp_nn_multicore.h>
#include "tensorflow/lite/micro/kernels/add.h"
#include <esp_heap_caps.h>
#include <algorithm>
#include <cstring>
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"

namespace lampnet_accel {
static size_t scratchBytes=0, internalBytes=0;
static void *scratch=nullptr, *internalScratch=nullptr;
static unsigned convNodes=0, depthwiseNodes=0, addNodes=0;
struct NodeData {
  tflite::OpDataConv quant;
  data_dims_t in, filter, out;
  conv_params_t conv;
  dw_conv_params_t depthwise;
};
static void *init(TfLiteContext *c,const char*,size_t) {
  auto *d=static_cast<NodeData*>(c->AllocatePersistentBuffer(c,sizeof(NodeData)));
  if(d)memset(d,0,sizeof(*d));
  return d;
}
template<bool Depthwise> static TfLiteStatus prepare(TfLiteContext *c,TfLiteNode *node) {
  auto *d=static_cast<NodeData*>(node->user_data);
  TF_LITE_ENSURE(c,d&&node->builtin_data);
  auto *mc=tflite::GetMicroContext(c);
  auto *in=mc->AllocateTempInputTensor(node,0);
  auto *filter=mc->AllocateTempInputTensor(node,1);
  auto *out=mc->AllocateTempOutputTensor(node,0);
  TF_LITE_ENSURE(c,in&&filter&&out);
  TF_LITE_ENSURE(c,in->type==kTfLiteInt8&&filter->type==kTfLiteInt8&&out->type==kTfLiteInt8);
  TF_LITE_ENSURE(c,in->dims->size==4&&filter->dims->size==4&&out->dims->size==4);
  TF_LITE_ENSURE(c,in->dims->data[0]==1&&out->dims->data[0]==1);
  d->in={in->dims->data[2],in->dims->data[1],in->dims->data[3],1};
  d->out={out->dims->data[2],out->dims->data[1],out->dims->data[3],1};
  d->filter={filter->dims->data[2],filter->dims->data[1],filter->dims->data[3],0};
  const auto *affine=static_cast<const TfLiteAffineQuantization*>(filter->quantization.params);
  TF_LITE_ENSURE(c,filter->quantization.type==kTfLiteAffineQuantization&&affine&&affine->zero_point);
  for(int i=0;i<affine->zero_point->size;++i)TF_LITE_ENSURE_EQ(c,affine->zero_point->data[i],0);
  auto &q=d->quant;
  q.per_channel_output_multiplier=static_cast<int32_t*>(c->AllocatePersistentBuffer(c,d->out.channels*sizeof(int32_t)));
  q.per_channel_output_shift=static_cast<int32_t*>(c->AllocatePersistentBuffer(c,d->out.channels*sizeof(int32_t)));
  TF_LITE_ENSURE(c,q.per_channel_output_multiplier&&q.per_channel_output_shift);
  int bytes=0;
  if constexpr(Depthwise) {
    const auto &p=*static_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);
    TF_LITE_ENSURE(c,p.dilation_width_factor==1&&p.dilation_height_factor==1);
    TF_LITE_ENSURE_EQ(c,d->out.channels,d->in.channels*p.depth_multiplier);
    TF_LITE_ENSURE_STATUS(tflite::CalculateOpDataDepthwiseConv(c,node,p,d->in.width,d->in.height,d->filter.width,d->filter.height,d->out.width,d->out.height,kTfLiteInt8,&q));
    d->depthwise={-q.input_zero_point,q.output_zero_point,p.depth_multiplier,{p.stride_width,p.stride_height},{q.padding.width,q.padding.height},{1,1},{q.output_activation_min,q.output_activation_max}};
    bytes=esp_nn_get_depthwise_conv_scratch_size(&d->in,&d->filter,&d->out,&d->depthwise);
    ++depthwiseNodes;
  } else {
    const auto &p=*static_cast<const TfLiteConvParams*>(node->builtin_data);
    TF_LITE_ENSURE(c,p.dilation_width_factor==1&&p.dilation_height_factor==1);
    TF_LITE_ENSURE_EQ(c,d->filter.channels,d->in.channels);
    TF_LITE_ENSURE_STATUS(tflite::CalculateOpDataConv(c,node,p,d->in.width,d->in.height,d->filter.width,d->filter.height,d->out.width,d->out.height,kTfLiteInt8,&q));
    d->conv={-q.input_zero_point,q.output_zero_point,{p.stride_width,p.stride_height},{q.padding.width,q.padding.height},{1,1},{q.output_activation_min,q.output_activation_max}};
    bytes=esp_nn_get_conv_scratch_size(&d->in,&d->filter,&d->out,&d->conv);
    ++convNodes;
  }
  TF_LITE_ENSURE(c,bytes>=0);
  scratchBytes=std::max(scratchBytes,size_t(bytes));
  mc->DeallocateTempTfLiteTensor(out);mc->DeallocateTempTfLiteTensor(filter);mc->DeallocateTempTfLiteTensor(in);
  return kTfLiteOk;
}
template<bool Depthwise> static TfLiteStatus eval(TfLiteContext *c,TfLiteNode *node) {
  auto *d=static_cast<NodeData*>(node->user_data);
  auto *in=tflite::micro::GetEvalInput(c,node,0);
  auto *filter=tflite::micro::GetEvalInput(c,node,1);
  auto *bias=node->inputs->size==3?tflite::micro::GetEvalInput(c,node,2):nullptr;
  auto *out=tflite::micro::GetEvalOutput(c,node,0);
  TF_LITE_ENSURE(c,scratch&&in&&filter&&out);
  quant_data_t quant={d->quant.per_channel_output_shift,d->quant.per_channel_output_multiplier};
  const int32_t *biasData=bias?tflite::micro::GetTensorData<int32_t>(bias):nullptr;
  if constexpr(Depthwise) {
    esp_nn_depthwise_conv_s8(&d->in,in->data.int8,&d->filter,filter->data.int8,biasData,&d->out,out->data.int8,&d->depthwise,&quant);
  } else {
    esp_nn_conv_s8(&d->in,in->data.int8,&d->filter,filter->data.int8,biasData,&d->out,out->data.int8,&d->conv,&quant);
  }
  return kTfLiteOk;
}
static bool allocateWorkspace() {
  // One serialized model worker shares workspace between layers. Larger
  // layers retain a full PSRAM buffer; kernels prefer internal RAM when fit.
  scratch=heap_caps_aligned_alloc(16,scratchBytes+16,MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
  if(!scratch)return false;
  for(size_t bytes:{size_t(32768),size_t(16384),size_t(8192)}) {
    internalScratch=heap_caps_aligned_alloc(16,bytes,MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT|MALLOC_CAP_DMA);
    if(internalScratch){internalBytes=bytes;break;}
  }
  esp_nn_set_conv_scratch_buf(scratch);esp_nn_set_depthwise_conv_scratch_buf(scratch);
  esp_nn_set_conv_preferred_scratch_buf_esp32s3(internalScratch,internalBytes);
  esp_nn_set_depthwise_conv_preferred_scratch_buf_esp32s3(internalScratch,internalBytes);
  return true;
}
struct AddData {tflite::OpDataAdd quant;int count;};
static void *addInit(TfLiteContext *c,const char*,size_t) {
  return c->AllocatePersistentBuffer(c,sizeof(AddData));
}
static TfLiteStatus addPrepare(TfLiteContext *c,TfLiteNode *node) {
  auto *d=static_cast<AddData*>(node->user_data);auto *mc=tflite::GetMicroContext(c);
  auto *a=mc->AllocateTempInputTensor(node,0),*b=mc->AllocateTempInputTensor(node,1),*out=mc->AllocateTempOutputTensor(node,0);
  TF_LITE_ENSURE(c,d&&a&&b&&out&&node->builtin_data);
  TF_LITE_ENSURE(c,a->type==kTfLiteInt8&&b->type==kTfLiteInt8&&out->type==kTfLiteInt8);
  TF_LITE_ENSURE_EQ(c,a->dims->size,b->dims->size);TF_LITE_ENSURE_EQ(c,a->dims->size,out->dims->size);
  d->count=1;
  for(int i=0;i<a->dims->size;++i){TF_LITE_ENSURE_EQ(c,a->dims->data[i],b->dims->data[i]);TF_LITE_ENSURE_EQ(c,a->dims->data[i],out->dims->data[i]);d->count*=out->dims->data[i];}
  TF_LITE_ENSURE_STATUS(tflite::CalculateOpDataAdd(c,static_cast<TfLiteAddParams*>(node->builtin_data),a,b,out,&d->quant));
  TF_LITE_ENSURE(c,!d->quant.requires_broadcast&&d->quant.input1_shift<=0&&d->quant.input2_shift<=0&&d->quant.output_shift<=0);
  mc->DeallocateTempTfLiteTensor(out);mc->DeallocateTempTfLiteTensor(b);mc->DeallocateTempTfLiteTensor(a);++addNodes;
  return kTfLiteOk;
}
static TfLiteStatus addEval(TfLiteContext *c,TfLiteNode *node) {
  auto &d=*static_cast<AddData*>(node->user_data);auto &q=d.quant;
  auto *a=tflite::micro::GetEvalInput(c,node,0),*b=tflite::micro::GetEvalInput(c,node,1);auto *out=tflite::micro::GetEvalOutput(c,node,0);
  esp_nn_add_elementwise_s8(a->data.int8,b->data.int8,q.input1_offset,q.input2_offset,q.input1_multiplier,q.input2_multiplier,q.input1_shift,q.input2_shift,q.left_shift,out->data.int8,q.output_offset,q.output_multiplier,q.output_shift,q.output_activation_min,q.output_activation_max,d.count);
  return kTfLiteOk;
}
static TfLiteRegistration add() {return tflite::micro::RegisterOp(addInit,addPrepare,addEval);}
static TfLiteRegistration conv() {return tflite::micro::RegisterOp(init,prepare<false>,eval<false>);}
static TfLiteRegistration depthwise() {return tflite::micro::RegisterOp(init,prepare<true>,eval<true>);}
} // namespace lampnet_accel
