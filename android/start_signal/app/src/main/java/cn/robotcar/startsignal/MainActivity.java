package cn.robotcar.startsignal;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.*;
import android.hardware.camera2.*;
import android.media.Image;
import android.media.ImageReader;
import android.os.*;
import android.util.Log;
import android.util.Size;
import android.view.*;
import android.widget.*;
import org.json.*;
import java.io.*;
import java.nio.ByteBuffer;
import java.util.*;

/** Local lamp sensor prototype; USB diagnostics are isolated in UsbCheckActivity. */
public final class MainActivity extends Activity {
    private LampModel model;
    private final SignalDetector signal=new SignalDetector();
    private HandlerThread thread;
    private Handler worker;
    private CameraDevice camera;
    private CameraCaptureSession session;
    private CaptureRequest.Builder request;
    private ImageReader reader;
    private volatile boolean resumed=false,recording=false;
    private boolean opening=false,exposureLocked=false,canLockAe=false,canLockAwb=false;
    private int rotation=90;
    private long lastAnalysis=0,lastSaved=0;
    private ImageView preview;
    private Overlay overlay;
    private TextView status,details;
    private Button recordButton;
    private volatile float roiWidth=.8f,roiCx=.5f,roiCy=.34f;
    private final Object recordLock=new Object();
    private File recordDir;
    private BufferedWriter events;
    private int frameIndex=0;
    private String cameraId="";

    @Override public void onCreate(Bundle state){
        super.onCreate(state);getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().setStatusBarColor(Color.rgb(15,23,42));getWindow().setNavigationBarColor(Color.rgb(15,23,42));
        LinearLayout root=new LinearLayout(this);root.setOrientation(LinearLayout.VERTICAL);root.setPadding(dp(16),dp(12),dp(16),dp(12));root.setBackgroundColor(Color.rgb(15,23,42));
        root.setOnApplyWindowInsetsListener((v,insets)->{v.setPadding(dp(16),dp(12)+insets.getSystemWindowInsetTop(),dp(16),dp(12)+insets.getSystemWindowInsetBottom());return insets;});
        TextView title=text("发车灯 · LampNet 0.1",23,Color.WHITE);root.addView(title);
        root.addView(text("识别原型 · 许可仅本机，USB 仅用于诊断",13,Color.rgb(148,163,184)));
        FrameLayout frame=new FrameLayout(this);preview=new ImageView(this);preview.setScaleType(ImageView.ScaleType.FIT_CENTER);
        frame.addView(preview,new FrameLayout.LayoutParams(-1,-1));overlay=new Overlay();frame.addView(overlay,new FrameLayout.LayoutParams(-1,-1));
        root.addView(frame,new LinearLayout.LayoutParams(-1,0,1));
        status=text("正在加载模型",19,Color.WHITE);root.addView(status);
        details=text("把亮灯完整放入青框，固定后按 ARM。",13,Color.rgb(203,213,225));details.setMinLines(3);root.addView(details);
        root.addView(text("拖动画面移动青框；滑动调整大小",12,Color.rgb(148,163,184)));
        SeekBar size=new SeekBar(this);size.setMax(80);size.setProgress(60);root.addView(size,new LinearLayout.LayoutParams(-1,dp(42)));
        size.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener(){public void onStartTrackingTouch(SeekBar b){}public void onStopTrackingTouch(SeekBar b){}public void onProgressChanged(SeekBar b,int v,boolean user){if(user){signal.stop();roiWidth=.2f+v/100f;overlay.invalidate();}}});
        LinearLayout buttons=new LinearLayout(this);
        Button arm=new Button(this);arm.setText("ARM 等灯灭");arm.setOnClickListener(v->{signal.arm();status.setText("寻找亮灯");});buttons.addView(arm,new LinearLayout.LayoutParams(0,dp(54),1));
        Button stop=new Button(this);stop.setText("STOP / 重置");stop.setTextColor(Color.rgb(239,68,68));stop.setOnClickListener(v->{signal.stop();status.setText("已重置本机许可");});buttons.addView(stop,new LinearLayout.LayoutParams(0,dp(54),1));
        recordButton=new Button(this);recordButton.setText("录制");recordButton.setOnClickListener(v->toggleRecording());buttons.addView(recordButton,new LinearLayout.LayoutParams(0,dp(54),1));root.addView(buttons);
        Button usb=new Button(this);usb.setText("USB / UNO 通信测试");
        usb.setOnClickListener(v->startActivity(new android.content.Intent(this,UsbCheckActivity.class)));
        root.addView(usb,new LinearLayout.LayoutParams(-1,dp(48)));
        setContentView(root);
        try{model=new LampModel(this);status.setText("模型已加载 · 等待相机");}catch(Exception e){showError("模型加载失败",e);}
        if(getIntent().getBooleanExtra("selftest",false) && model!=null)new Thread(this::selftest,"lamp-selftest").start();
        handleDiagnosticIntent(getIntent());
        if(checkSelfPermission(Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED)requestPermissions(new String[]{Manifest.permission.CAMERA},7);
    }
    private int dp(int n){return Math.round(n*getResources().getDisplayMetrics().density);}
    @Override protected void onNewIntent(android.content.Intent intent){super.onNewIntent(intent);setIntent(intent);handleDiagnosticIntent(intent);if(intent.getBooleanExtra("selftest",false)&&model!=null)new Thread(this::selftest,"lamp-selftest").start();}
    private void handleDiagnosticIntent(android.content.Intent intent){
        if((getApplicationInfo().flags&android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE)==0)return;
        long duration=Math.min(15000,Math.max(0,intent.getLongExtra("record_for_ms",0)));
        if(duration>0&&!recording){toggleRecording();File ownDir=recordDir;new Handler(Looper.getMainLooper()).postDelayed(()->{if(recordDir==ownDir)stopRecording();},duration);}
    }
    private TextView text(String s,int sp,int color){TextView t=new TextView(this);t.setText(s);t.setTextSize(sp);t.setTextColor(color);t.setPadding(dp(2),dp(4),dp(2),dp(4));return t;}
    private void showError(String label,Exception e){Log.e("RobotLamp",label,e);runOnUiThread(()->{status.setText(label);details.setText(e.getClass().getSimpleName()+": "+e.getMessage());});}
    @Override public void onResume(){super.onResume();resumed=true;thread=new HandlerThread("lamp-camera");thread.start();worker=new Handler(thread.getLooper());openCamera();}
    @Override public void onPause(){
        resumed=false;signal.gap();stopRecording();
        if(session!=null){session.close();session=null;}if(camera!=null){camera.close();camera=null;}
        if(reader!=null){reader.close();reader=null;}request=null;opening=false;
        if(thread!=null){thread.quitSafely();try{thread.join(500);}catch(InterruptedException e){Thread.currentThread().interrupt();}thread=null;}worker=null;
        super.onPause();
    }
    @Override public void onDestroy(){if(model!=null)model.close();super.onDestroy();}
    @Override public void onRequestPermissionsResult(int code,String[] permissions,int[] results){super.onRequestPermissionsResult(code,permissions,results);if(code==7&&results.length>0&&results[0]==PackageManager.PERMISSION_GRANTED)openCamera();else status.setText("请允许相机权限以识别信号灯");}
    private void openCamera(){
        if(!resumed||opening||camera!=null||model==null||worker==null||checkSelfPermission(Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED)return;
        opening=true;final Handler handler=worker;
        try{
            CameraManager manager=(CameraManager)getSystemService(CAMERA_SERVICE);CameraCharacteristics chosen=null;
            for(String id:manager.getCameraIdList()){
                CameraCharacteristics c=manager.getCameraCharacteristics(id);
                if(Objects.equals(c.get(CameraCharacteristics.LENS_FACING),CameraCharacteristics.LENS_FACING_BACK)){cameraId=id;chosen=c;break;}
            }
            if(chosen==null)throw new IllegalStateException("No rear camera");
            rotation=chosen.get(CameraCharacteristics.SENSOR_ORIENTATION);
            canLockAe=Boolean.TRUE.equals(chosen.get(CameraCharacteristics.CONTROL_AE_LOCK_AVAILABLE));
            canLockAwb=Boolean.TRUE.equals(chosen.get(CameraCharacteristics.CONTROL_AWB_LOCK_AVAILABLE));
            Size[] sizes=chosen.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP).getOutputSizes(ImageFormat.YUV_420_888);
            Size size=Arrays.stream(sizes).min(Comparator.comparingDouble(s->Math.abs(s.getWidth()-640)+Math.abs(s.getHeight()-480))).orElseThrow(()->new IllegalStateException("No YUV camera size"));
            reader=ImageReader.newInstance(size.getWidth(),size.getHeight(),ImageFormat.YUV_420_888,3);reader.setOnImageAvailableListener(this::onImage,handler);
            manager.openCamera(cameraId,new CameraDevice.StateCallback(){
                public void onOpened(CameraDevice device){opening=false;if(!resumed){device.close();return;}camera=device;createSession(handler);}
                public void onDisconnected(CameraDevice device){device.close();camera=null;opening=false;signal.gap();runOnUiThread(()->status.setText("相机中断，等待恢复"));}
                public void onError(CameraDevice device,int error){device.close();camera=null;opening=false;signal.gap();runOnUiThread(()->status.setText("相机错误 "+error));}
            },handler);
        }catch(Exception e){opening=false;showError("相机打开失败",e);}
    }
    private void createSession(Handler handler){try{
        Surface surface=reader.getSurface();request=camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);request.addTarget(surface);
        request.set(CaptureRequest.CONTROL_AF_MODE,CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
        exposureLocked=false;
        camera.createCaptureSession(Collections.singletonList(surface),new CameraCaptureSession.StateCallback(){
            public void onConfigured(CameraCaptureSession s){if(!resumed){s.close();return;}session=s;try{s.setRepeatingRequest(request.build(),null,handler);}catch(Exception e){showError("相机预览失败",e);}}
            public void onConfigureFailed(CameraCaptureSession s){runOnUiThread(()->status.setText("相机配置失败"));}
        },handler);
    }catch(Exception e){showError("相机配置失败",e);}}
    private void updateLocks(boolean locked){if(locked==exposureLocked||session==null||request==null)return;try{
        if(canLockAe)request.set(CaptureRequest.CONTROL_AE_LOCK,locked);
        if(canLockAwb)request.set(CaptureRequest.CONTROL_AWB_LOCK,locked);
        session.setRepeatingRequest(request.build(),null,worker);exposureLocked=locked;
    }catch(Exception e){Log.w("RobotLamp","AE/AWB lock unavailable",e);}}
    private static int clamp(int n){return Math.max(0,Math.min(255,n));}
    private Bitmap toBitmap(Image image){
        int w=image.getWidth(),h=image.getHeight();Image.Plane[] p=image.getPlanes();
        ByteBuffer yy=p[0].getBuffer(),uu=p[1].getBuffer(),vv=p[2].getBuffer();int[] out=new int[w*h];
        int yp=p[0].getPixelStride(),yr=p[0].getRowStride(),up=p[1].getPixelStride(),ur=p[1].getRowStride(),vp=p[2].getPixelStride(),vr=p[2].getRowStride();
        for(int y=0;y<h;y++)for(int x=0;x<w;x++){
            int Y=(yy.get(y*yr+x*yp)&255)-16,U=(uu.get((y/2)*ur+(x/2)*up)&255)-128,V=(vv.get((y/2)*vr+(x/2)*vp)&255)-128;
            Y=Math.max(0,Y);int r=clamp((298*Y+409*V+128)>>8),g=clamp((298*Y-100*U-208*V+128)>>8),b=clamp((298*Y+516*U+128)>>8);out[y*w+x]=0xff000000|(r<<16)|(g<<8)|b;
        }
        Bitmap raw=Bitmap.createBitmap(out,w,h,Bitmap.Config.ARGB_8888);if(rotation==0)return raw;
        Matrix matrix=new Matrix();matrix.postRotate(rotation);Bitmap upright=Bitmap.createBitmap(raw,0,0,w,h,matrix,true);raw.recycle();return upright;
    }
    private Rect roi(int w,int h){int rw=Math.max(32,Math.min(w,(int)(w*roiWidth))),rh=Math.min(h,(int)(rw*.8f));int x=Math.max(0,Math.min(w-rw,(int)(roiCx*w-rw/2f))),y=Math.max(0,Math.min(h-rh,(int)(roiCy*h-rh/2f)));return new Rect(x,y,x+rw,y+rh);}
    private void onImage(ImageReader source){
        Image image=null;try{
            image=source.acquireLatestImage();if(image==null)return;
            long now=SystemClock.elapsedRealtime();if(!resumed||now-lastAnalysis<100)return;lastAnalysis=now;
            long sensorTimestamp=image.getTimestamp();Bitmap full=toBitmap(image);image.close();image=null;
            Rect region=roi(full.getWidth(),full.getHeight());Bitmap cropped=Bitmap.createBitmap(full,region.left,region.top,region.width(),region.height());
            Bitmap input=Bitmap.createScaledBitmap(cropped,LampModel.SIZE,LampModel.SIZE,true);if(input!=cropped)cropped.recycle();
            LampModel.Result result=model.run(input,region.width()/(float)region.height());input.recycle();boolean event=signal.update(result.grey,result.mask,result.valid(),now);
            boolean[] locked=signal.lockedMask();updateLocks(locked!=null);
            JSONObject state=new JSONObject();state.put("t_ms",now);state.put("camera_timestamp_ns",sensorTimestamp);state.put("phase",signal.phase);state.put("reason",signal.reason);state.put("permission",signal.permission());state.put("event",event);state.put("lamp_mean",signal.lamp);state.put("drop",signal.drop);state.put("dark_fraction",signal.darkFraction);state.put("background_residual",signal.backgroundResidual);state.put("model_confidence",result.confidence);state.put("model_ms",result.inferenceMs);state.put("mask_cells",result.area);state.put("roi",new JSONArray(new int[]{region.left,region.top,region.width(),region.height()}));
            if(event)Log.i("RobotLamp","LOCAL_START_PERMISSION "+state);
            saveFrame(full,state,now);
            String phase=signal.phase,reason=signal.reason;boolean permitted=signal.permission();
            String info=signal.armed()||permitted?String.format(Locale.ROOT,"%s\n灯面 %.0f · 下降 %.0f · 熄灯像素 %.0f%% · 模型 %.1f ms\n仅本机识别，尚未连接小车",reason,signal.lamp,signal.drop,signal.darkFraction*100,result.inferenceMs):String.format(Locale.ROOT,"%s\n模型 %.1f ms · 仅本机识别，尚未连接小车",result.valid()?"找到灯面候选；亮灯固定后按 ARM":"把亮灯完整放入青框，固定后按 ARM",result.inferenceMs);
            runOnUiThread(()->{if(!resumed){full.recycle();return;}preview.setImageBitmap(full);overlay.set(full.getWidth(),full.getHeight(),region,result,locked);status.setText(phase);status.setTextColor(permitted?Color.rgb(74,222,128):Color.WHITE);details.setText(info);});
        }catch(Exception e){if(resumed)showError("识别处理中断",e);signal.gap();}finally{if(image!=null)image.close();}
    }
    private void toggleRecording(){if(recording){stopRecording();return;}synchronized(recordLock){try{
        recordDir=new File(getExternalFilesDir(null),"recordings/"+System.currentTimeMillis());if(!recordDir.mkdirs())throw new IOException("Cannot create recording directory");
        events=new BufferedWriter(new FileWriter(new File(recordDir,"events.jsonl")));frameIndex=0;lastSaved=0;recording=true;recordButton.setText("停止录制");
    }catch(Exception e){showError("无法录制",e);}}}
    private void stopRecording(){synchronized(recordLock){recording=false;if(events!=null){try{events.close();}catch(IOException ignored){}events=null;}}runOnUiThread(()->{if(recordButton!=null)recordButton.setText("录制");});}
    private void saveFrame(Bitmap image,JSONObject state,long now)throws Exception{synchronized(recordLock){if(!recording)return;
        if(now-lastSaved>=500){String filename=String.format(Locale.ROOT,"%06d.jpg",frameIndex++);try(FileOutputStream out=new FileOutputStream(new File(recordDir,filename))){image.compress(Bitmap.CompressFormat.JPEG,88,out);}state.put("file",filename);lastSaved=now;}
        events.write(state.toString());events.newLine();events.flush();
    }}
    private void selftest(){try{
        JSONArray tests=new JSONArray(readAsset("golden_tests.json"));JSONArray results=new JSONArray();long begin=System.nanoTime();
        for(int i=0;i<tests.length();i++){
            JSONObject t=tests.getJSONObject(i);Bitmap original;
            try(InputStream stream=getAssets().open(t.getString("asset"))){original=BitmapFactory.decodeStream(stream);}
            JSONArray r=t.getJSONArray("crop");Bitmap crop=Bitmap.createBitmap(original,r.getInt(0),r.getInt(1),r.getInt(2),r.getInt(3));Bitmap input=Bitmap.createScaledBitmap(crop,160,160,true);
            LampModel.Result result=model.run(input,crop.getWidth()/(float)crop.getHeight());JSONObject row=new JSONObject();row.put("asset",t.getString("asset"));row.put("valid",result.valid());row.put("area",result.area);row.put("confidence",result.confidence);row.put("ms",result.inferenceMs);results.put(row);
            input.recycle();if(input!=crop)crop.recycle();if(crop!=original)original.recycle();
        }
        JSONObject report=new JSONObject();report.put("model","LampNet160");report.put("results",results);report.put("elapsed_ms",(System.nanoTime()-begin)/1e6);report.put("device",Build.MODEL);
        File dir=getExternalFilesDir(null);try(FileWriter out=new FileWriter(new File(dir,"selftest.json"))){out.write(report.toString(2));}Log.i("RobotLamp","SELFTEST "+report);
    }catch(Exception e){Log.e("RobotLamp","SELFTEST FAILED",e);}}
    private String readAsset(String name)throws IOException{try(InputStream stream=getAssets().open(name);ByteArrayOutputStream out=new ByteArrayOutputStream()){byte[] buf=new byte[8192];int n;while((n=stream.read(buf))!=-1)out.write(buf,0,n);return out.toString("UTF-8");}}
    private final class Overlay extends View {
        final Paint paint=new Paint(Paint.ANTI_ALIAS_FLAG);int iw=1,ih=1;Rect region=new Rect();LampModel.Result result;boolean[] mask;
        Overlay(){super(MainActivity.this);setLayerType(View.LAYER_TYPE_SOFTWARE,null);}
        void set(int w,int h,Rect r,LampModel.Result target,boolean[] locked){iw=w;ih=h;region=new Rect(r);result=target;mask=locked!=null?locked:(target.valid()?target.mask:null);invalidate();}
        float scale(){return Math.min(getWidth()/(float)iw,getHeight()/(float)ih);}
        @Override protected void onDraw(Canvas c){super.onDraw(c);if(result==null)return;float s=scale(),dx=(getWidth()-iw*s)/2,dy=(getHeight()-ih*s)/2;c.save();c.translate(dx,dy);c.scale(s,s);
            paint.setStyle(Paint.Style.STROKE);paint.setStrokeWidth(3/s);paint.setColor(Color.CYAN);c.drawRect(region,paint);
            if(mask!=null){paint.setStyle(Paint.Style.FILL);paint.setColor(Color.argb(65,255,210,0));for(int y=0;y<40;y++)for(int x=0;x<40;x++)if(mask[y*40+x])c.drawRect(region.left+x*region.width()/40f,region.top+y*region.height()/40f,region.left+(x+1)*region.width()/40f,region.top+(y+1)*region.height()/40f,paint);}
            if(result.valid()){RectF b=result.box;paint.setStyle(Paint.Style.STROKE);paint.setColor(Color.YELLOW);c.drawRect(region.left+b.left*region.width(),region.top+b.top*region.height(),region.left+b.right*region.width(),region.top+b.bottom*region.height(),paint);}c.restore();}
        @Override public boolean onTouchEvent(MotionEvent e){if(e.getAction()==MotionEvent.ACTION_DOWN||e.getAction()==MotionEvent.ACTION_MOVE){float s=scale();if(s<=0)return true;roiCx=Math.max(0,Math.min(1,(e.getX()-(getWidth()-iw*s)/2)/s/iw));roiCy=Math.max(0,Math.min(1,(e.getY()-(getHeight()-ih*s)/2)/s/ih));signal.stop();return true;}return true;}
    }
}
