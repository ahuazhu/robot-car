package cn.robotcar.startsignal;
import java.util.Arrays;
public class SignalDetectorCheck {
    static boolean[] mask=new boolean[1600];
    static float[] frame(float panel,float bg){float[] g=new float[1600];Arrays.fill(g,bg);for(int i=0;i<g.length;i++)if(mask[i])g[i]=panel;return g;}
    static void check(boolean b,String s){if(!b)throw new AssertionError(s);}
    static void lock(SignalDetector d){d.arm();for(long t=1000;t<=1400;t+=100)d.update(frame(210,80),mask,true,t);check(d.lockedMask()!=null,"must lock stable ON");}
    public static void main(String[] args){
        for(int y=12;y<24;y++)for(int x=14;x<26;x++)mask[y*40+x]=true;
        SignalDetector d=new SignalDetector();lock(d);
        for(long t=1500;t<2200;t+=100)d.update(frame(210,80),mask,true,t);
        check(!d.permission(),"steady ON must not fire");
        for(long t=2200;t<2600;t+=100)d.update(frame(70,80),mask,true,t);
        check(!d.permission(),"must wait 400ms");d.update(frame(70,80),mask,true,2600);
        check(d.permission(),"ON to OFF must fire");d.gap();check(d.permission(),"gap must preserve permit");
        d.stop();check(!d.permission(),"STOP clears permit");
        lock(d);for(long t=1500;t<2500;t+=100)d.update(frame(170,40),mask,true,t);
        check(!d.permission(),"global offset must not fire");
        d.arm();float[] gradient=frame(210,80);for(int i=0;i<gradient.length;i++)if(!mask[i])gradient[i]=30+(i%40)*4;
        for(long t=1000;t<=1400;t+=100)d.update(gradient,mask,true,t);
        float[] gain=gradient.clone();for(int i=0;i<gain.length;i++)gain[i]=gain[i]*.65f+12;
        for(long t=1500;t<=2300;t+=100)d.update(gain,mask,true,t);
        check(!d.permission(),"global gain and offset must not fire");
        lock(d);d.update(frame(70,80),mask,true,1500);d.update(frame(70,80),mask,true,1700);
        d.update(frame(70,80),mask,true,2500);check(!d.permission(),"gap cannot complete OFF");
        d.stop();for(long t=3000;t<4000;t+=100)d.update(frame(70,80),mask,true,t);
        check(!d.permission(),"unarmed cannot fire");
        d.arm();float[] sparse=frame(25,80);for(int y=12;y<24;y+=2)for(int x=14;x<26;x+=2)sparse[y*40+x]=210;
        for(long t=1000;t<=1400;t+=100)d.update(sparse,mask,true,t);
        check(d.lockedMask()!=null,"sparse LED matrix must lock");
        int core=0;for(boolean b:d.lockedMask())if(b)core++;check(core==36,"must only freeze emitting cores");
        for(long t=1500;t<=2000;t+=100)d.update(frame(25,80),mask,true,t);
        check(d.permission(),"sparse LED OFF must fire");
        // STOP must invalidate the cached background for a new panel position.
        d.stop();Arrays.fill(mask,false);
        for(int y=4;y<16;y++)for(int x=4;x<16;x++)mask[y*40+x]=true;
        lock(d);
        // Losing the locator after locking must not change the frozen region.
        for(long t=1500;t<=2000;t+=100)d.update(frame(70,80),null,false,t);
        check(d.permission(),"new panel geometry must replace the previous exposure mask");
        System.out.println("SignalDetector checks passed");
    }
}
