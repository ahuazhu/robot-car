package cn.robotcar.startsignal;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import org.tensorflow.lite.Interpreter;
import java.io.InputStream;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/** Model locates the panel; SignalDetector alone decides the temporal event. */
public final class LampModel implements AutoCloseable {
    public static final int SIZE=160, GRID=40;
    private final Interpreter interpreter;
    private final ByteBuffer input=ByteBuffer.allocateDirect(SIZE*SIZE*3*4).order(ByteOrder.nativeOrder());
    private final float[][][][] output=new float[1][GRID][GRID][1];
    private final int[] pixels=new int[SIZE*SIZE];
    public static final class Result {
        public final boolean[] mask;
        public final float[] grey;
        public final RectF box;
        public final float confidence, inferenceMs, sourceAspect;
        public final int area;
        Result(boolean[] m,float[] g,RectF b,float c,float ms,int a,float aspect) {mask=m;grey=g;box=b;confidence=c;inferenceMs=ms;area=a;sourceAspect=aspect;}
        public boolean valid(){
            if(area<6 || area>=GRID*GRID*.65 || confidence<.65f || box==null)return false;
            float aspect=box.width()*sourceAspect/box.height();
            float fill=area/(box.width()*box.height()*GRID*GRID);
            return aspect>=.5f && aspect<=1.9f && fill>=.45f;
        }
    }
    public LampModel(Context context) throws Exception {
        byte[] blob;
        try(InputStream stream=context.getAssets().open("lamp_model.tflite");ByteArrayOutputStream out=new ByteArrayOutputStream()){
            byte[] buffer=new byte[8192];int n;while((n=stream.read(buffer))!=-1)out.write(buffer,0,n);blob=out.toByteArray();
        }
        ByteBuffer model=ByteBuffer.allocateDirect(blob.length).order(ByteOrder.nativeOrder());model.put(blob);model.rewind();
        interpreter=new Interpreter(model,new Interpreter.Options().setNumThreads(2));
        if(!Arrays.equals(interpreter.getInputTensor(0).shape(),new int[]{1,SIZE,SIZE,3}) ||
           !Arrays.equals(interpreter.getOutputTensor(0).shape(),new int[]{1,GRID,GRID,1}))throw new IllegalStateException("Unexpected model tensor shapes");
    }
    public synchronized Result run(Bitmap bitmap) {return run(bitmap,1);}
    public synchronized Result run(Bitmap bitmap,float sourceAspect) {
        if(bitmap.getWidth()!=SIZE || bitmap.getHeight()!=SIZE)throw new IllegalArgumentException("Model input must be 160x160");
        bitmap.getPixels(pixels,0,SIZE,0,0,SIZE,SIZE);input.rewind();
        for(int rgb:pixels){input.putFloat(((rgb>>16)&255)/127.5f-1);input.putFloat(((rgb>>8)&255)/127.5f-1);input.putFloat((rgb&255)/127.5f-1);}
        input.rewind();
        long started=System.nanoTime();interpreter.run(input,output);
        float ms=(System.nanoTime()-started)/1e6f;
        float[] grey=new float[GRID*GRID];boolean[] visited=new boolean[GRID*GRID];
        for(int y=0;y<GRID;y++)for(int x=0;x<GRID;x++){
            float value=0;
            for(int dy=0;dy<4;dy++)for(int dx=0;dx<4;dx++){
                int rgb=pixels[(y*4+dy)*SIZE+x*4+dx];
                value+=Math.max((rgb>>16)&255,Math.max((rgb>>8)&255,rgb&255));
            }
            // HSV value is color-independent: a saturated blue/red LED must
            // not fail an ON threshold just because its luminance is low.
            grey[y*GRID+x]=value/16;
        }
        int[] queue=new int[GRID*GRID];boolean[] best=new boolean[GRID*GRID];int bestSize=0,second=0;
        float confidence=0;RectF box=null;
        for(int start=0;start<GRID*GRID;start++) {
            if(visited[start] || output[0][start/GRID][start%GRID][0]<.55f)continue;
            int head=0,tail=1;queue[0]=start;visited[start]=true;
            int xmin=GRID,ymin=GRID,xmax=0,ymax=0;float sum=0;
            while(head<tail){
                int k=queue[head++],x=k%GRID,y=k/GRID;
                xmin=Math.min(xmin,x);xmax=Math.max(xmax,x);ymin=Math.min(ymin,y);ymax=Math.max(ymax,y);sum+=output[0][y][x][0];
                for(int d:new int[]{-1,1,-GRID,GRID}){
                    int j=k+d;if(j<0||j>=GRID*GRID||Math.abs(j%GRID-x)+Math.abs(j/GRID-y)!=1||visited[j]||output[0][j/GRID][j%GRID][0]<.55f)continue;
                    visited[j]=true;queue[tail++]=j;
                }
            }
            if(tail>bestSize){
                second=bestSize;bestSize=tail;Arrays.fill(best,false);for(int i=0;i<tail;i++)best[queue[i]]=true;
                confidence=sum/tail;box=new RectF(xmin/(float)GRID,ymin/(float)GRID,(xmax+1f)/GRID,(ymax+1f)/GRID);
            }else second=Math.max(second,tail);
        }
        // Similar competing panels require repositioning instead of guessing.
        if(second>bestSize*.65f && second>=6)confidence=0;
        return new Result(best,grey,box,confidence,ms,bestSize,sourceAspect);
    }
    @Override public synchronized void close(){interpreter.close();}
}
