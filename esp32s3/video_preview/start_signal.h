#pragma once
// Shared by the ESP32 firmware and native replay; no camera/network dependencies.
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

namespace start_signal {
struct Region { int x=0,y=0,w=0,h=0,count=0,direction=0; float strength=0; int threshold=0,color=-1,seed=0; };

class Detector {
 public:
  static constexpr int X=80,Y=0,RW=160,RH=128,W=RW/2,H=RH/2,N=W*H;
  static constexpr int ALIGNMENT_WIDTH=128,ALIGNMENT_HEIGHT=112;
  static constexpr unsigned LOCK_FRAMES=4,LOCK_MS=300,CONFIRM_FRAMES=3,CONFIRM_MS=400;
  enum Phase { IDLE,SEARCH,ACQUIRE,ON,CONFIRM_OFF,UNCERTAIN,FIRED };
  bool armed=false,confirmed=false,baseline=false;
  uint32_t sequence=0,hits=0,elapsed=0,targetHits=0;
  Region target,region;
  Phase phase=IDLE;
  float contrast=0,baselineContrast=0,noise=0;
  float backgroundChangeL=0,backgroundResidualL=0,darkenedFraction=0,requiredDrop=0,exposureGain=1;
  unsigned maskPixels=0;
  const char *rejectionReason="DISARMED";

  Detector(){
    for(int i=0;i<256;++i){float s=i/255.f;linear[i]=s<=.04045f?s/12.92f:powf((s+.055f)/1.055f,2.4f);}
    for(int i=0;i<1024;++i){float y=i/1023.f;float f=y>.00885645f?cbrtf(y):7.787037f*y+16.f/116.f;int l=int((116.f*f-16.f)*2.55f+.5f);lightness[i]=l<0?0:l>255?255:l;}
  }
  void cancel(){armed=confirmed=baseline=false;target=region=Region();targetHits=0;hits=elapsed=0;haveLast=false;contrast=baselineContrast=noise=0;backgroundChangeL=backgroundResidualL=darkenedFraction=requiredDrop=0;exposureGain=1;maskPixels=0;rejectionReason="DISARMED";phase=IDLE;}
  void arm(){cancel();armed=true;phase=SEARCH;rejectionReason="WAIT_LAMP_FACE";}
  void copyMask(uint8_t *out) const {if(baseline)memcpy(out,lamp,N);else memset(out,0,N);}
  uint32_t eventAge(uint32_t now) const {return hits||confirmed?uint32_t(now-offSince):0;}
  const char *phaseName() const {
    switch(phase){case SEARCH:return "WAIT_TARGET";case ACQUIRE:return "LOCK_TARGET";case ON:return "WAIT_OFF";case CONFIRM_OFF:return "CONFIRM_OFF";case UNCERTAIN:return "TARGET_UNCERTAIN";case FIRED:return "SIGNAL_READY";default:return "DISARMED";}
  }
  void process(const uint8_t *rgb565,uint32_t now,int roiY=Y){
    ++sequence;
    if(!armed||confirmed)return;
    prepare(rgb565,roiY);
    if(haveLast&&uint32_t(now-last)>750){baseline=false;targetHits=0;hits=elapsed=0;phase=SEARCH;}
    haveLast=true;last=now;
    if(!baseline){acquire(now);return;}
    compare(now);
  }

 private:
  // Workspace belongs to the detector, never the FreeRTOS camera task stack.
  uint8_t current[N],reference[N],rgb[N*3],colourMask[N],seen[N],lamp[N],ring[N],background[N],component[N];
  uint16_t queue[N];float linear[256];uint8_t lightness[1024];
  uint32_t last=0,lockSince=0,offSince=0;bool haveLast=false;
  float minContrast=0,maxContrast=0,meanContrast=0;
  uint16_t exposureHistogram[16][256];

  static float overlap(const Region&a,const Region&b){
    int x0=a.x>b.x?a.x:b.x,y0=a.y>b.y?a.y:b.y;
    int x1=a.x+a.w<b.x+b.w?a.x+a.w:b.x+b.w,y1=a.y+a.h<b.y+b.h?a.y+a.h:b.y+b.h;
    int area=(x1>x0&&y1>y0)?(x1-x0)*(y1-y0):0;
    int total=a.w*a.h+b.w*b.h-area;return total?float(area)/total:0;
  }
  static int minimum(int a,int b){return a<b?a:b;}
  static int maximum(int a,int b){return a>b?a:b;}
  // Chromaticity only locates the ON face. White/clipped lamps also have a
  // luminance-only path; no competition-specific red/green assumption.
  uint8_t classifyColour(int i) const {
    int r=rgb[i*3],g=rgb[i*3+1],b=rgb[i*3+2];
    int hi=maximum(r,maximum(g,b)),lo=minimum(r,minimum(g,b)),chroma=hi-lo;
    uint8_t membership=0;
    if(chroma*255<=hi*8)membership|=1;
    if(chroma*255<=hi*24)membership|=2;
    if(chroma*255<hi*48||!chroma)return membership;
    int hue=hi==r?256*(g-b)/chroma:hi==g?512+256*(b-r)/chroma:1024+256*(r-g)/chroma;
    if(hue<0)hue+=1536;
    for(int color=2;color<8;++color){int distance=abs(hue-(color-2)*256);distance=minimum(distance,1536-distance);if(distance<=160)membership|=1<<color;}
    return membership;
  }
  bool eligible(int i,int threshold,int color) const {
    return current[i]>=threshold&&(color<0||(colourMask[i]&(1<<color)));
  }
  void masks(const Region&r){
    memset(lamp,0,N);memset(ring,0,N);memset(background,0,N);
    // Reconstruct the connected component, then remove its bloom boundary.
    memset(component,0,N);int head=0,tail=0;queue[tail++]=r.seed;component[r.seed]=1;
    while(head<tail){int k=queue[head++],x=k%W,y=k/W;
      for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){int a=x+dx,b=y+dy;
        if(a<r.x||a>=r.x+r.w||b<r.y||b>=r.y+r.h)continue;
        int q=b*W+a;if(!component[q]&&eligible(q,r.threshold,r.color)){component[q]=1;queue[tail++]=q;}
      }
    }
    unsigned count=0;
    for(int y=r.y+1;y<r.y+r.h-1;++y)for(int x=r.x+1;x<r.x+r.w-1;++x){
      bool inner=true;for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx)if(!component[(y+dy)*W+x+dx])inner=false;
      if(inner){lamp[y*W+x]=1;++count;}
    }
    // Tiny but complete faces still need a usable core.
    if(count<9){memcpy(lamp,component,N);count=tail;}
    maskPixels=count;
    for(int y=0;y<H;++y)for(int x=0;x<W;++x){
      bool near=x>=r.x-5&&x<r.x+r.w+5&&y>=r.y-5&&y<r.y+r.h+5;
      if(near&&(x<r.x-1||x>r.x+r.w||y<r.y-1||y>r.y+r.h))ring[y*W+x]=1;
      // Exposure reference is spread across the search window, outside bloom.
      if(x<r.x-3||x>=r.x+r.w+3||y<r.y-3||y>=r.y+r.h+3)background[y*W+x]=1;
    }
  }
  float mean(const uint8_t *pixels,const uint8_t *selection) const {
    unsigned sum=0,count=0;for(int i=0;i<N;++i)if(selection[i]){sum+=pixels[i];++count;}return count?float(sum)/count:0;
  }
  Region locate(){
    unsigned histogram[256]={},sum=0;for(int i=0;i<N;++i)++histogram[current[i]];
    int median=0;for(;median<255;++median){sum+=histogram[median];if(sum>=N/2)break;}
    Region candidates[16];float scores[16]={};int candidateCount=0;
    bool preferredFound=false,outerOverflow=false;
    // A bright lamp may merge with bloom/window glare at a low threshold.
    // Inspect several brightness levels and group nested views of one face.
    for(int threshold=minimum(248,maximum(40,median+12));;threshold=minimum(248,threshold+24)){
    for(int color=-1;color<8;++color){
    memset(seen,0,N);
    for(int i=0;i<N;++i)if(eligible(i,threshold,color)&&!seen[i]){
      int head=0,tail=0,minx=W,miny=H,maxx=0,maxy=0,count=0;
      queue[tail++]=i;seen[i]=1;
      while(head<tail){int k=queue[head++],x=k%W,y=k/W;++count;
        minx=minimum(minx,x);maxx=maximum(maxx,x);miny=minimum(miny,y);maxy=maximum(maxy,y);
        for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){int a=x+dx,b=y+dy;
          if(a>=0&&a<W&&b>=0&&b<H){int q=b*W+a;if(!seen[q]&&eligible(q,threshold,color)){seen[q]=1;queue[tail++]=q;}}
        }
      }
      int w=maxx-minx+1,h=maxy-miny+1;
      // A planar square projects to a compact quadrilateral; reject clipped,
      // sparse and very elongated bright regions, including window edges.
      if(w<4||h<4||w>h*2.5f||h>w*2.5f||count>N/2||count*10<w*h*6||minx<1||miny<1||maxx>=W-1||maxy>=H-1)continue;
      int ax=X+(RW-ALIGNMENT_WIDTH)/2,ay=Y+(RH-ALIGNMENT_HEIGHT)/2;
      bool preferred=X+minx*2>=ax+2&&Y+miny*2>=ay+2
          &&X+(maxx+1)*2<=ax+ALIGNMENT_WIDTH-2
          &&Y+(maxy+1)*2<=ay+ALIGNMENT_HEIGHT-2;
      if((preferredFound||outerOverflow)&&!preferred)continue;
      Region r;r.x=minx;r.y=miny;r.w=w;r.h=h;r.count=count;r.direction=-1;r.threshold=threshold;r.color=color;r.seed=i;
      masks(r);float c=mean(current,lamp)-mean(current,ring);if(c<14)continue;
      float cx=minx+w*.5f-W*.5f,cy=miny+h*.5f-H*.5f;
      // Keep alignment preference in sensor pixels when the search box grows.
      // 46.4 native pixels retains the original centre weighting strength.
      float score=c*sqrtf(float(count))/(1.f+4.f*(cx*cx+cy*cy)/(46.4f*46.4f));
      if(targetHits&&overlap(target,r)>=.65f)score*=1.5f;
      // Preserve the original alignment area as the first choice. Newly
      // visible background must not displace a valid central lamp; the larger
      // window remains a fallback when the original area has no candidate.
      if(preferred&&!preferredFound){candidateCount=0;preferredFound=true;}
      r.strength=c;
      int group=-1;
      for(int j=0;j<candidateCount;++j)if(overlap(candidates[j],r)>=.4f){group=j;break;}
      if(group>=0){if(score>scores[group]){candidates[group]=r;scores[group]=score;}}
      else if(candidateCount<16){candidates[candidateCount]=r;scores[candidateCount++]=score;}
      else if(preferredFound)return Region(); // Ambiguous central targets.
      else outerOverflow=true;
    }
    } // color partitions
    if(threshold==248)break;
    }
    if(!preferredFound&&outerOverflow)return Region();
    Region best;float bestScore=0,secondScore=0;
    for(int j=0;j<candidateCount;++j){float score=scores[j];if(score>bestScore){secondScore=bestScore;bestScore=score;best=candidates[j];}else if(score>secondScore)secondScore=score;}
    // Ambiguous targets must be reframed rather than silently choosing a person.
    return secondScore>bestScore*.85f?Region():best;
  }
  void acquire(uint32_t now){
    backgroundChangeL=backgroundResidualL=darkenedFraction=requiredDrop=0;
    Region found=locate();
    if(!found.count){target=Region();maskPixels=0;targetHits=0;phase=SEARCH;rejectionReason="NO_LAMP_CANDIDATE";return;}
    float c=found.strength;
    if(!targetHits||overlap(target,found)<.65f){target=found;targetHits=1;lockSince=now;minContrast=maxContrast=meanContrast=c;phase=ACQUIRE;rejectionReason="ACQUIRING_LAMP";return;}
    ++targetHits;minContrast=fminf(minContrast,c);maxContrast=fmaxf(maxContrast,c);meanContrast+=(c-meanContrast)/targetHits;
    if(targetHits<LOCK_FRAMES||uint32_t(now-lockSince)<LOCK_MS){phase=ACQUIRE;rejectionReason="ACQUIRING_LAMP";return;}
    noise=maxContrast-minContrast;
    if(noise>maximum(8,int(meanContrast*.25f))){targetHits=0;phase=UNCERTAIN;rejectionReason="UNSTABLE_BRIGHT_BASELINE";return;}
    target=found;masks(target);memcpy(reference,current,N);baselineContrast=mean(current,lamp)-mean(current,ring);
    contrast=baselineContrast;baseline=true;hits=elapsed=0;phase=ON;rejectionReason="WAIT_OFF";
  }
  static int quantile(const uint16_t *hist,int size,unsigned count){
    unsigned sum=0;for(int i=0;i<size;++i){sum+=hist[i];if(sum>=(count+1)/2)return i;}return size-1;
  }
  // Fit only when the background has enough independent brightness levels.
  // Flat backgrounds cannot identify gain AND offset from one observation.
  bool fitExposure(float &gain,float &offset){
    memset(exposureHistogram,0,sizeof(exposureHistogram));
    unsigned counts[16]={};float sums[16]={},x[16]={},y[16]={};bool usable[16]={};
    for(int i=0;i<N;++i)if(background[i]&&reference[i]>=32&&reference[i]<=245){
      int bin=reference[i]/16;++counts[bin];sums[bin]+=reference[i];++exposureHistogram[bin][current[i]];
    }
    int bins=0;float low=255,high=0;
    for(int j=0;j<16;++j)if(counts[j]>=24){usable[j]=true;++bins;x[j]=sums[j]/counts[j];y[j]=quantile(exposureHistogram[j],256,counts[j]);low=fminf(low,x[j]);high=fmaxf(high,x[j]);}
    if(bins<3||high-low<32)return false;
    unsigned best=0;float bestGain=1,bestOffset=0;
    for(int a=0;a<16;++a)if(usable[a])for(int b=a+1;b<16;++b)if(usable[b]&&x[b]-x[a]>=24){
      float g=(y[b]-y[a])/(x[b]-x[a]),o=y[a]-g*x[a];if(g<.4f||g>1.6f||fabsf(o)>48)continue;
      unsigned votes=0;for(int j=0;j<16;++j)if(usable[j]&&fabsf(g*x[j]+o-y[j])<=6)++votes;
      if(votes>best){best=votes;bestGain=g;bestOffset=o;}
    }
    if(best<3||best*5<unsigned(bins)*4)return false;
    float sx=0,sy=0,sxx=0,sxy=0,n=0;
    for(int j=0;j<16;++j)if(usable[j]&&fabsf(bestGain*x[j]+bestOffset-y[j])<=6){sx+=x[j];sy+=y[j];sxx+=x[j]*x[j];sxy+=x[j]*y[j];++n;}
    float denominator=n*sxx-sx*sx;if(denominator<1)return false;
    gain=(n*sxy-sx*sy)/denominator;offset=(sy-gain*sx)/n;
    return gain>=.4f&&gain<=1.6f&&fabsf(offset)<=48;
  }
  void compare(uint32_t now){
    uint16_t deltaHistogram[511]={},ratioHistogram[513]={},residualHistogram[256]={};unsigned samples=0,ratioSamples=0;
    for(int i=0;i<N;++i)if(background[i]&&reference[i]<=245){
      ++deltaHistogram[int(current[i])-reference[i]+255];
      if(reference[i]>=32){++ratioHistogram[minimum(512,int(current[i])*128/reference[i])];++ratioSamples;}++samples;
    }
    if(samples<32){hits=elapsed=0;phase=UNCERTAIN;rejectionReason="NO_BACKGROUND_REFERENCE";return;}
    float delta=quantile(deltaHistogram,511,samples)-255;
    float gain=ratioSamples>=32?quantile(ratioHistogram,513,ratioSamples)/128.f:1.f;
    backgroundChangeL=delta;exposureGain=gain;
    float fittedGain=1,fittedOffset=0;bool fitted=fitExposure(fittedGain,fittedOffset);
    unsigned stable=0,changed=0,lampCount=0,ringCount=0,newBright=0;float dropSum=0;
    // A single flat background cannot distinguish additive light change from
    // gain. Require actual darkening AND darkening under both exposure
    // models, rather than inventing a fit or firing on a brighter background.
    for(int i=0;i<N;++i){
      float expected=fminf(reference[i],fminf(reference[i]+delta,reference[i]*gain));
      if(fitted)expected=fminf(expected,reference[i]*fittedGain+fittedOffset);
      expected=fminf(255.f,fmaxf(0.f,expected));
      float difference=expected-current[i];
      if(background[i]&&reference[i]<=245){
        // Light spill need not affect a nearby white beam and the ceiling
        // equally. Count reference patches compatible with identity OR a
        // global exposure model; do not mistake this for geometric motion.
        float error=fminf(fabsf(float(current[i])-reference[i]),fminf(fabsf(float(current[i])-reference[i]-delta),fabsf(float(current[i])-reference[i]*gain)));
        int residual=minimum(255,int(error+.5f));++residualHistogram[residual];if(residual<=18)++stable;
      }
      if(ring[i]){++ringCount;float brightest=fmaxf(reference[i],fmaxf(reference[i]+delta,reference[i]*gain));if(current[i]>brightest+20)++newBright;}
      if(lamp[i]){++lampCount;dropSum+=difference;if(difference>=8)++changed;}
    }
    backgroundResidualL=quantile(residualHistogram,256,samples);
    darkenedFraction=lampCount?float(changed)/lampCount:0;
    contrast=mean(current,lamp)-mean(current,ring);
    if(stable*10<samples*6||backgroundResidualL>14||(ringCount&&newBright*5>ringCount)){hits=elapsed=0;phase=UNCERTAIN;rejectionReason="BACKGROUND_STRUCTURE_CHANGED";return;}
    float threshold=fmaxf(8.f,fmaxf(noise*3.f,baselineContrast*.10f));
    requiredDrop=threshold;float drop=lampCount?dropSum/lampCount:0;
    bool off=drop>=threshold&&lampCount&&changed*10>=lampCount*6;
    if(!off){hits=elapsed=0;phase=drop<threshold*.4f?ON:UNCERTAIN;rejectionReason=drop<threshold?"LAMP_NOT_DARK_ENOUGH":"TOO_FEW_DARKENED_PIXELS";return;}
    if(!hits)offSince=now;
    ++hits;elapsed=uint32_t(now-offSince);phase=CONFIRM_OFF;rejectionReason="CONFIRMING_OFF";region=target;region.count=changed;region.strength=drop;
    if(hits>=CONFIRM_FRAMES&&elapsed>=CONFIRM_MS){confirmed=true;phase=FIRED;rejectionReason="SIGNAL_CONFIRMED";}
  }
  void prepare(const uint8_t *pixels,int roiY){
    for(int y=0;y<H;++y)for(int x=0;x<W;++x){int sums[3]={};
      for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){int at=((roiY+y*2+dy)*320+X+x*2+dx)*2;unsigned v=(unsigned(pixels[at])<<8)|pixels[at+1];
        sums[0]+=((v>>11)&31)*255/31;sums[1]+=((v>>5)&63)*255/63;sums[2]+=(v&31)*255/31;}
      for(int c=0;c<3;++c)rgb[(y*W+x)*3+c]=(sums[c]+2)/4;
      colourMask[y*W+x]=classifyColour(y*W+x);
    }
    for(int y=0;y<H;++y)for(int x=0;x<W;++x){int sums[3]={};
      for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){int a=x+dx,b=y+dy;if(a<0)a=1;if(a>=W)a=W-2;if(b<0)b=1;if(b>=H)b=H-2;int weight=(dx==0?2:1)*(dy==0?2:1);
        for(int c=0;c<3;++c)sums[c]+=weight*rgb[(b*W+a)*3+c];}
      float luminance=.2126729f*linear[(sums[0]+8)/16]+.7151522f*linear[(sums[1]+8)/16]+.0721750f*linear[(sums[2]+8)/16];
      int index=int(luminance*1023+.5f);current[y*W+x]=lightness[minimum(1023,maximum(0,index))];
    }
  }
};
}
