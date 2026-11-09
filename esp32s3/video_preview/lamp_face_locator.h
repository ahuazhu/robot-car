#pragma once
// Camera-independent, bounded workspace; shared by firmware and native replay.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

namespace lamp_face {
constexpr int SIDE=40, PIXELS=SIDE*SIDE;
enum Method {NONE,COARSE,MODEL_CORE,PHOTOMETRIC};
struct Box {int x=0,y=0,w=0,h=0;};
struct Result {
  bool valid=false,coarseValid=false;Method method=NONE;
  Box box,coarseBox;unsigned area=0,threshold=141;
  float confidence=0;
};
struct Component {
  int x=0,y=0,w=0,h=0;unsigned area=0,second=0,sum=0;
  bool ambiguous()const{return second>=6&&second>area*.65f;}
  bool valid(float aspect)const{return area>=6&&area<1040&&sum>=area*256*.65f&&w*aspect>=h*.5f&&w*aspect<=h*1.9f&&area>=w*h*.45f&&!ambiguous();}
};
class Locator {
 public:
  uint8_t mask[PIXELS]={};
  void view(int x,int y,int w,int h){viewX=x;viewY=y;viewW=w;viewH=h;}
  Result locate(const uint8_t *prob,const uint8_t *rgb) {
    memset(mask,0,PIXELS);
    Component broad=extract(prob,141,nullptr,nullptr,coarse);
    Result r;r.coarseValid=broad.valid(float(viewW)/viewH);r.coarseBox=box(broad);
    if(!broad.area)return r;
    use(r,broad,COARSE,141,coarse);
    if(broad.ambiguous())return r;
    unsigned peak=0;for(int i=0;i<PIXELS;++i)if(coarse[i]&&prob[i]>peak)peak=prob[i];
    unsigned threshold=peak>145?peak-4:141;if(threshold>251)threshold=251;
    Component inner=extract(prob,threshold,coarse,nullptr,core);
    if(!inner.valid(float(viewW)/viewH))return r; // Preserve weak/small coarse candidates.
    use(r,inner,MODEL_CORE,threshold,core);
    if(!rgb){margin(r,inner);return r;}
    memset(histogram,0,sizeof(histogram));
    for(int i=0;i<PIXELS;++i)if(core[i]) {
      int a=rgb[i*3],b=rgb[i*3+1],c=rgb[i*3+2],hi=max(a,max(b,c)),lo=min(a,min(b,c));
      ++histogram[0][a];++histogram[1][b];++histogram[2][c];++histogram[3][hi];++histogram[4][lo];++histogram[5][hi-lo];
    }
    float med[6];for(int k=0;k<6;++k)med[k]=median(histogram[k],inner.area);
    float total=med[0]+med[1]+med[2]+1,colour[3]={med[0]/total,med[1]/total,med[2]/total};
    for(int i=0;i<PIXELS;++i) {
      photo[i]=0;if(!coarse[i])continue;
      int a=rgb[i*3],b=rgb[i*3+1],c=rgb[i*3+2],hi=max(a,max(b,c)),lo=min(a,min(b,c));
      if(med[5]<=40)photo[i]=lo>=med[4]-12;
      else {float sum=a+b+c+1.f;photo[i]=fabsf(a/sum-colour[0])+fabsf(b/sum-colour[1])+fabsf(c/sum-colour[2])<=.16f&&hi>=med[3]-12;}
    }
    if(min(inner.w,inner.h)>=8)opening();
    Component refined=extract(prob,141,nullptr,photo,temporary);
    unsigned common=0;for(int i=0;i<PIXELS;++i)if(core[i]&&temporary[i])++common;
    if(refined.valid(float(viewW)/viewH)&&common>=6&&common>=inner.area*.5f) {
      use(r,refined,PHOTOMETRIC,threshold,temporary);margin(r,refined);
    }else margin(r,inner);
    return r;
  }
 private:
  int viewX=80,viewY=0,viewW=160,viewH=128;
  uint8_t coarse[PIXELS],core[PIXELS],photo[PIXELS],temporary[PIXELS],seen[PIXELS];
  uint16_t queue[PIXELS],histogram[6][256];
  static int min(int a,int b){return a<b?a:b;}
  static int max(int a,int b){return a>b?a:b;}
  Box box(const Component &c)const{return {viewX+c.x*viewW/40,viewY+c.y*viewH/40,c.w*viewW/40,c.h*viewH/40};}
  void use(Result &r,const Component &c,Method m,unsigned t,const uint8_t *selection) {
    r.valid=c.valid(float(viewW)/viewH);r.box=box(c);r.area=c.area;r.confidence=c.area?c.sum/(c.area*256.f):0;r.method=m;r.threshold=t;memcpy(mask,selection,PIXELS);
  }
  void margin(Result &r,const Component &c) {
    if(min(c.w,c.h)<8)return;
    int x=max(0,c.x-1),y=max(0,c.y-1),right=min(40,c.x+c.w+1),bottom=min(40,c.y+c.h+1);
    r.box={viewX+x*viewW/40,viewY+y*viewH/40,(right-x)*viewW/40,(bottom-y)*viewH/40};
  }
  static float median(const uint16_t *hist,unsigned n) {
    unsigned sum=0;int a=-1,b=255;
    for(int i=0;i<256;++i){sum+=hist[i];if(a<0&&sum>(n-1)/2)a=i;if(sum>n/2){b=i;break;}}
    return (a+b)*.5f;
  }
  Component extract(const uint8_t *prob,unsigned threshold,const uint8_t *scope,const uint8_t *allowed,uint8_t *out) {
    memset(seen,0,PIXELS);memset(out,0,PIXELS);Component best;
    for(int i=0;i<PIXELS;++i) {
      if(seen[i]||prob[i]<threshold||(scope&&!scope[i])||(allowed&&!allowed[i]))continue;
      unsigned head=0,tail=1,sum=0;queue[0]=i;seen[i]=1;int x0=40,y0=40,x1=0,y1=0;
      while(head<tail) {
        int k=queue[head++],x=k%40,y=k/40;x0=min(x0,x);y0=min(y0,y);x1=max(x1,x);y1=max(y1,y);sum+=prob[k];
        const int direction[4]={-1,1,-40,40};
        for(int d:direction){int j=k+d;if(j<0||j>=PIXELS||abs(j%40-x)+abs(j/40-y)!=1||seen[j]||prob[j]<threshold||(scope&&!scope[j])||(allowed&&!allowed[j]))continue;seen[j]=1;queue[tail++]=j;}
      }
      if(tail>best.area){unsigned second=best.area;best={x0,y0,x1-x0+1,y1-y0+1,tail,second,sum};memset(out,0,PIXELS);for(unsigned j=0;j<tail;++j)out[queue[j]]=1;}
      else if(tail>best.second)best.second=tail;
    }
    return best;
  }
  void opening() {
    // Erode then dilate; outside-image convention matches OpenCV's defaults.
    for(int y=0;y<40;++y)for(int x=0;x<40;++x){bool yes=true;for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){int a=x+dx,b=y+dy;if(a>=0&&a<40&&b>=0&&b<40&&!photo[b*40+a])yes=false;}temporary[y*40+x]=yes;}
    for(int y=0;y<40;++y)for(int x=0;x<40;++x){bool yes=false;for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){int a=x+dx,b=y+dy;if(a>=0&&a<40&&b>=0&&b<40&&temporary[b*40+a])yes=true;}photo[y*40+x]=yes;}
  }
};
inline const char *methodName(Method m){return m==PHOTOMETRIC?"MODEL_COLOUR_CORE":m==MODEL_CORE?"MODEL_CONFIDENT_CORE":m==COARSE?"COARSE_FALLBACK":"NO_CANDIDATE";}
} // namespace lamp_face
