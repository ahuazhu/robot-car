package cn.robotcar.startsignal;

import java.util.Arrays;

/** Pure Java state machine. Permission is local and latched until explicit STOP.
 * No class here sends commands to a motor or opens a network connection.
 */
public final class SignalDetector {
    public static final int GRID=40;
    private boolean armed=false,permitted=false;
    private boolean[] mask,candidateMask,backgroundMask;
    private float[] reference;
    private int stable=0,hits=0;
    private long stableSince=0,offSince=0,lastFrame=0;
    public String phase="未 ARM", reason="请把亮灯放入青框";
    public float lamp=0,drop=0,darkFraction=0,backgroundResidual=0;
    public synchronized void arm(){stop();armed=true;phase="寻找亮灯";reason="固定灯和相机，等待亮灯稳定";}
    public synchronized void stop(){armed=permitted=false;mask=candidateMask=backgroundMask=null;reference=null;stable=hits=0;lastFrame=offSince=0;phase="未 ARM";reason="请把亮灯放入青框";}
    public synchronized boolean permission(){return permitted;}
    public synchronized boolean armed(){return armed;}
    public synchronized int hits(){return hits;}
    public synchronized boolean[] lockedMask(){return mask==null?null:mask.clone();}
    public synchronized void gap(){if(permitted)return;mask=null;reference=null;candidateMask=backgroundMask=null;stable=hits=0;phase=armed?"寻找亮灯":"未 ARM";reason="画面中断，重新确认亮灯";}
    private static float mean(float[] values,boolean[] mask){float s=0;int n=0;for(int i=0;i<values.length;i++)if(mask[i]){s+=values[i];n++;}return n==0?0:s/n;}
    private static float overlap(boolean[] a,boolean[] b){int inter=0,union=0;for(int i=0;i<a.length;i++){if(a[i]&&b[i])inter++;if(a[i]||b[i])union++;}return union==0?0:inter/(float)union;}
    private void resetOff(String why){hits=0;offSince=0;darkFraction=0;reason=why;}
    private static boolean[] exposureBackground(boolean[] panel){
        // The locked panel never moves within one trial. Exclude its bloom
        // once rather than rebuilding this fixed geometry on every frame.
        boolean[] background=new boolean[panel.length];Arrays.fill(background,true);
        for(int i=0;i<panel.length;i++)if(panel[i]){
            int x=i%GRID,y=i/GRID;
            for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++)
                if(x+dx>=0&&x+dx<GRID&&y+dy>=0&&y+dy<GRID)background[(y+dy)*GRID+x+dx]=false;
        }
        return background;
    }
    public synchronized boolean update(float[] grey,boolean[] target,boolean valid,long now){
        if(permitted){phase="已确认发车信号";return false;}
        if(lastFrame>0 && now-lastFrame>600)gap();lastFrame=now;
        if(!armed)return false;
        if(mask==null){
            if(!valid||target==null){stable=0;candidateMask=null;phase="寻找亮灯";reason="未找到可信灯面";return false;}
            // A real panel can be a sparse LED matrix. Freeze its emitting
            // cores, not the black gaps between LEDs or the numeric timer.
            float[] values=new float[grey.length];int count=0;
            int px0=GRID,py0=GRID,px1=0,py1=0;
            for(int i=0;i<grey.length;i++)if(target[i]){values[count++]=grey[i];px0=Math.min(px0,i%GRID);px1=Math.max(px1,i%GRID);py0=Math.min(py0,i/GRID);py1=Math.max(py1,i/GRID);}
            if(count<4){stable=0;return false;}
            Arrays.sort(values,0,count);float high=values[Math.min(count-1,(int)(count*.9f))];
            if(high<85){stable=0;reason="已找到灯面，等待亮灯";return false;}
            boolean[] emitters=new boolean[grey.length];int emitted=0,ex0=GRID,ey0=GRID,ex1=0,ey1=0;
            float threshold=Math.max(55,high*.55f);
            for(int i=0;i<grey.length;i++)if(target[i]&&grey[i]>=threshold){emitters[i]=true;emitted++;ex0=Math.min(ex0,i%GRID);ex1=Math.max(ex1,i%GRID);ey0=Math.min(ey0,i/GRID);ey1=Math.max(ey1,i/GRID);}
            if(emitted<Math.max(4,count*.08f)||(ex1-ex0+1)<(px1-px0+1)*.45f||(ey1-ey0+1)<(py1-py0+1)*.45f){stable=0;reason="亮点集中在边缘，等待灯面点亮";return false;}
            lamp=mean(grey,emitters);
            if(candidateMask==null||overlap(candidateMask,target)<.6f){stable=0;stableSince=now;}
            candidateMask=target.clone();stable++;phase="确认亮灯";reason="亮灯稳定 "+stable+"/4 帧";
            if(stable>=4 && now-stableSince>=300){
                mask=emitters;backgroundMask=exposureBackground(target);reference=grey.clone();phase="等待熄灯";reason="灯面已锁定";
            }
            return false;
        }
        phase="等待熄灯";lamp=mean(grey,mask);
        float[] deltas=new float[grey.length];int n=0;
        double sx=0,sy=0,sxx=0,sxy=0;
        for(int i=0;i<grey.length;i++)if(backgroundMask[i]){
            deltas[n++]=grey[i]-reference[i];
            sx+=reference[i];sy+=grey[i];sxx+=reference[i]*reference[i];sxy+=reference[i]*grey[i];
        }
        if(n<40){resetOff("灯面太大，四周留出背景");return false;}
        Arrays.sort(deltas,0,n);float offset=deltas[n/2],gain=1;
        double variance=sxx/n-(sx/n)*(sx/n);
        if(variance>225){
            gain=(float)((sxy-sx*sy/n)/(sxx-sx*sx/n));
            if(gain<.5f||gain>1.8f){resetOff("背景曝光变化过大");return false;}
            int j=0;for(int i=0;i<grey.length;i++)if(backgroundMask[i])deltas[j++]=grey[i]-reference[i]*gain;
            Arrays.sort(deltas,0,n);offset=deltas[n/2];
        }else if(Math.abs(offset)>12){resetOff("背景缺少层次，等待曝光稳定");return false;}
        float residual=0;for(int i=0;i<grey.length;i++)if(backgroundMask[i])residual+=Math.abs(grey[i]-reference[i]*gain-offset);
        backgroundResidual=residual/n;
        if(backgroundResidual>22){resetOff("画面移动或背景变化，保持固定");return false;}
        int dark=0,total=0;float sum=0;
        for(int i=0;i<grey.length;i++)if(mask[i]){
            float difference=reference[i]*gain+offset-grey[i];sum+=difference;total++;
            if(difference>=Math.max(10,reference[i]*.15f))dark++;
        }
        drop=sum/Math.max(1,total);darkFraction=dark/(float)Math.max(1,total);
        if(darkFraction>=.6f && drop>=10){
            if(hits==0)offSince=now;hits++;reason="熄灯确认 "+hits+"/3 帧";
            if(hits>=3 && now-offSince>=400){permitted=true;armed=false;phase="已确认发车信号";reason="本机许可已锁存；STOP 后清除";return true;}
        }else resetOff("灯面已锁定，等待亮度下降");
        return false;
    }
}
