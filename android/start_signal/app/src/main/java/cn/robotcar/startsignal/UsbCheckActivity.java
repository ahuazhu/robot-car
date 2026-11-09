package cn.robotcar.startsignal;

import android.app.Activity;
import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.ApplicationInfo;
import android.graphics.Color;
import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbDeviceConnection;
import android.hardware.usb.UsbManager;
import android.os.Build;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import com.hoho.android.usbserial.driver.UsbSerialDriver;
import com.hoho.android.usbserial.driver.UsbSerialPort;
import com.hoho.android.usbserial.driver.UsbSerialProber;

import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Bounded UNO USB diagnostic. The only application data ever transmitted is one 'i'. */
public final class UsbCheckActivity extends Activity {
    public static final String EXTRA_RUN_TEST = "run_test";
    private static final String ACTION_PERMISSION = "cn.robotcar.startsignal.USB_CHECK_PERMISSION";
    private static final long TEST_LIMIT_MS = 14000;
    private static final int READ_TIMEOUT_MS = 100;
    private final Object ioLock = new Object();
    private final Object protocolLock = new Object();
    private final Object reportLock = new Object();
    private volatile boolean destroyed;
    private volatile boolean running;
    private boolean permissionPending;
    private boolean probeRequested;
    private volatile long generation;
    private long activeGeneration = -1;
    private UsbManager manager;
    private UsbDevice selectedDevice;
    private UsbSerialPort activePort;
    private UsbDeviceConnection activeConnection;
    private volatile UsbCheckProtocol protocol;
    private TextView statusText;
    private TextView serialText;
    private Button retryButton;
    private File reportDirectory;
    private final StringBuilder recent = new StringBuilder();
    private int receivedBytes;
    private int discardedBytes;
    private int txBytes;
    private boolean dtrEnabled;
    private String runId = "";

    private final BroadcastReceiver usbReceiver = new BroadcastReceiver() {
        @Override public void onReceive(Context context, Intent intent) {
            UsbDevice device = intent.getParcelableExtra(UsbManager.EXTRA_DEVICE);
            if (ACTION_PERMISSION.equals(intent.getAction())) {
                if (!permissionPending || destroyed || !sameDevice(device, selectedDevice)) return;
                permissionPending = false;
                if (intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)
                        && manager.hasPermission(selectedDevice)) {
                    beginSerial(generation);
                } else {
                    show("未授予 USB 权限；没有打开串口，也没有发送数据。");
                    persist(generation, "PERMISSION_DENIED", "usb_permission_denied");
                    retryButton.setEnabled(true);
                }
            } else if (UsbManager.ACTION_USB_DEVICE_DETACHED.equals(intent.getAction())
                    && sameDevice(device, selectedDevice)) {
                cancel("usb_detached");
            }
        }
    };

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        manager = (UsbManager) getSystemService(USB_SERVICE);
        File external = getExternalFilesDir(null);
        reportDirectory = new File(external == null ? getFilesDir() : external, "usb-check");
        reportDirectory.mkdirs();
        buildUi();
        IntentFilter filter = new IntentFilter(ACTION_PERMISSION);
        filter.addAction(UsbManager.ACTION_USB_DEVICE_DETACHED);
        if (Build.VERSION.SDK_INT >= 33) registerReceiver(usbReceiver, filter, Context.RECEIVER_NOT_EXPORTED);
        else registerReceiver(usbReceiver, filter);
        startAttempt(allowedProbe(getIntent()));
    }

    @Override protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        if (running || permissionPending) {
            show("检查正在进行；先停止检查，再重新打开诊断。");
            return;
        }
        startAttempt(allowedProbe(intent));
    }

    private boolean allowedProbe(Intent intent) {
        return (getApplicationInfo().flags & ApplicationInfo.FLAG_DEBUGGABLE) != 0
                && intent != null && intent.getBooleanExtra(EXTRA_RUN_TEST, false);
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(24, 36, 24, 24);
        root.setBackgroundColor(Color.rgb(15, 23, 42));
        TextView title = text("UNO · USB 通信检查", 23);
        root.addView(title);
        root.addView(text("请先在 Maix 停车。打开 USB/DTR 可能使 UNO 复位。\n默认只读；诊断不发送 ARM、运动或清除急停命令。", 14));
        statusText = text("准备检查", 18);
        statusText.setContentDescription("usb-check-status");
        root.addView(statusText);
        LinearLayout buttons = new LinearLayout(this);
        retryButton = new Button(this);
        retryButton.setText("重新只读检查");
        retryButton.setOnClickListener(v -> startAttempt(false));
        buttons.addView(retryButton, new LinearLayout.LayoutParams(0, -2, 1));
        Button stop = new Button(this);
        stop.setText("停止检查");
        stop.setOnClickListener(v -> cancel("cancelled_by_user"));
        buttons.addView(stop, new LinearLayout.LayoutParams(0, -2, 1));
        root.addView(buttons);
        root.addView(text("结果目录：" + reportDirectory.getAbsolutePath(), 12));
        ScrollView scroll = new ScrollView(this);
        serialText = text("", 12);
        serialText.setTextIsSelectable(true);
        scroll.addView(serialText);
        root.addView(scroll, new LinearLayout.LayoutParams(-1, 0, 1));
        setContentView(root);
    }

    private TextView text(String value, int size) {
        TextView view = new TextView(this);
        view.setText(value); view.setTextSize(size); view.setTextColor(Color.WHITE);
        view.setPadding(0, 8, 0, 8);
        return view;
    }

    private void show(String value) {
        runOnUiThread(() -> { if (!destroyed) statusText.setText(value); });
    }

    private static boolean sameDevice(UsbDevice first, UsbDevice second) {
        return first != null && second != null && first.getDeviceName().equals(second.getDeviceName());
    }

    private void startAttempt(boolean test) {
        if (destroyed || running || permissionPending) return;
        generation++;
        probeRequested = test;
        protocol = null;
        receivedBytes = discardedBytes = txBytes = 0;
        dtrEnabled = false;
        runId = Instant.now().toString();
        recent.setLength(0); serialText.setText("");
        retryButton.setEnabled(false);
        try (FileOutputStream ignored = new FileOutputStream(new File(reportDirectory, "latest-serial.log"), false)) {
            // Each attempt owns a new raw log; a previous report is never evidence.
        } catch (IOException exc) {
            show("无法建立诊断日志：" + exc.getClass().getSimpleName());
            retryButton.setEnabled(true);
            return;
        }
        List<UsbSerialDriver> drivers;
        try {
            drivers = UsbSerialProber.getDefaultProber().findAllDrivers(manager);
        } catch (Exception exc) {
            persist(generation, "ERROR", "usb_enumeration_failed:" + exc.getClass().getSimpleName());
            retryButton.setEnabled(true);
            return;
        }
        UsbSerialDriver chosen = null;
        for (UsbSerialDriver driver : drivers) {
            UsbDevice device = driver.getDevice();
            if (device.getVendorId() == 0x2341 && device.getProductId() == 0x0043) chosen = driver;
        }
        if (chosen == null && drivers.size() == 1) chosen = drivers.get(0);
        if (chosen == null) {
            selectedDevice = null;
            String error = drivers.isEmpty() ? "no_supported_usb_serial_device" : "multiple_usb_serial_devices";
            show(drivers.isEmpty() ? "未发现支持的 USB 串口；检查 OTG 数据线和 UNO 供电。" : "发现多个串口，未自动选择或发送。请只连接 UNO。");
            persist(generation, "NO_DEVICE", error);
            retryButton.setEnabled(true);
            return;
        }
        selectedDevice = chosen.getDevice();
        show(test ? "安全往返诊断：先读取两条新鲜停车状态，再仅发送一次 i。" : "只读检查：不会发送串口应用数据。");
        if (manager.hasPermission(selectedDevice)) {
            beginSerial(generation);
        } else {
            permissionPending = true;
            persist(generation, "WAIT_PERMISSION", "");
            Intent intent = new Intent(ACTION_PERMISSION).setPackage(getPackageName());
            int flags = PendingIntent.FLAG_UPDATE_CURRENT;
            if (Build.VERSION.SDK_INT >= 31) flags |= PendingIntent.FLAG_MUTABLE;
            PendingIntent permission = PendingIntent.getBroadcast(this, 341, intent, flags);
            try {
                manager.requestPermission(selectedDevice, permission);
            } catch (Exception exc) {
                permissionPending = false;
                persist(generation, "ERROR", "usb_permission_failed:" + exc.getClass().getSimpleName());
                retryButton.setEnabled(true);
            }
        }
    }

    private void beginSerial(long ownGeneration) {
        if (destroyed || running || ownGeneration != generation || selectedDevice == null) return;
        running = true;
        UsbDevice device = selectedDevice;
        new Thread(() -> runSerial(ownGeneration, device), "uno-usb-check").start();
    }

    private boolean current(long ownGeneration) {
        return !destroyed && generation == ownGeneration;
    }

    private void runSerial(long ownGeneration, UsbDevice device) {
        FileOutputStream raw = null;
        UsbCheckProtocol ownProtocol = null;
        try {
            UsbSerialDriver driver = UsbSerialProber.getDefaultProber().probeDevice(device);
            if (driver == null || driver.getPorts().isEmpty()) throw new IOException("usb_driver_unavailable");
            UsbDeviceConnection connection = manager.openDevice(device);
            if (connection == null) throw new IOException("usb_open_failed");
            synchronized (ioLock) {
                if (!current(ownGeneration)) { connection.close(); return; }
                activeGeneration = ownGeneration;
                activeConnection = connection;
            }
            UsbSerialPort port = driver.getPorts().get(0);
            port.open(connection);
            synchronized (ioLock) {
                if (!current(ownGeneration)) { port.close(); return; }
                activePort = port;
            }
            port.setParameters(115200, 8, UsbSerialPort.STOPBITS_1, UsbSerialPort.PARITY_NONE);
            try { port.setDTR(true); dtrEnabled = true; } catch (UnsupportedOperationException ignored) { }
            long opened = SystemClock.elapsedRealtime();
            ownProtocol = new UsbCheckProtocol(probeRequested, opened);
            synchronized (protocolLock) { protocol = ownProtocol; }
            raw = new FileOutputStream(new File(reportDirectory, "latest-serial.log"), true);
            byte[] buffer = new byte[512];
            // Ignore an initial buffered fragment/reset banner for test evidence.
            long discardUntil = opened + 200;
            while (current(ownGeneration) && SystemClock.elapsedRealtime() < discardUntil) {
                int count = port.read(buffer, 25);
                if (count > 0) { raw.write(buffer, 0, count); receivedBytes += count; discardedBytes += count; }
            }
            StringBuilder line = new StringBuilder();
            boolean[] dropping = {false};
            boolean sent = false;
            while (current(ownGeneration) && SystemClock.elapsedRealtime() - opened < TEST_LIMIT_MS) {
                int count = port.read(buffer, READ_TIMEOUT_MS);
                if (count > 0) {
                    raw.write(buffer, 0, count); receivedBytes += count;
                    parse(buffer, count, line, dropping, ownProtocol);
                    raw.flush();
                    persist(ownGeneration, "", "");
                }
                synchronized (protocolLock) {
                    if (ownProtocol.isComplete() || ownProtocol.isFailed()) break;
                }
                boolean send;
                synchronized (protocolLock) { send = !sent && ownProtocol.shouldSendProbe(SystemClock.elapsedRealtime()); }
                if (send) {
                    // Drain to a quiet boundary before TX. Parse any new DRIVE
                    // row so a late armed/fault/nonzero state still cancels TX.
                    boolean quiet = false;
                    long drainUntil = SystemClock.elapsedRealtime() + 100;
                    while (current(ownGeneration) && SystemClock.elapsedRealtime() < drainUntil) {
                        int pending = port.read(buffer, 20);
                        if (pending == 0) { quiet = true; break; }
                        raw.write(buffer, 0, pending); receivedBytes += pending;
                        parse(buffer, pending, line, dropping, ownProtocol);
                    }
                    synchronized (protocolLock) {
                        if (!quiet || line.length() != 0 || dropping[0]) ownProtocol.fail("input_not_quiet_before_probe");
                        send = quiet && line.length() == 0 && !dropping[0]
                                && ownProtocol.shouldSendProbe(SystemClock.elapsedRealtime());
                    }
                    if (send) {
                        // Serialize the final gate and fixed write against closing.
                        // Cancellation never leaves a queued write behind it.
                        synchronized (ioLock) {
                            synchronized (protocolLock) {
                                if (current(ownGeneration) && activePort == port
                                        && ownProtocol.shouldSendProbe(SystemClock.elapsedRealtime())) {
                                    // The only TX: no ARM, PWM or estop clear.
                                    port.write(new byte[]{'i'}, 250);
                                    sent = true; txBytes = 1;
                                    ownProtocol.markProbeSent(SystemClock.elapsedRealtime());
                                }
                            }
                        }
                        persist(ownGeneration, "", "");
                    }
                }
                if (receivedBytes > 1024 * 1024) {
                    synchronized (protocolLock) { ownProtocol.fail("serial_input_limit"); }
                    break;
                }
            }
            synchronized (protocolLock) {
                if (!ownProtocol.isComplete() && !ownProtocol.isFailed()) {
                    ownProtocol.fail(current(ownGeneration) ? "diagnostic_timeout" : "diagnostic_cancelled");
                }
            }
        } catch (Exception exc) {
            if (ownProtocol != null) {
                synchronized (protocolLock) { ownProtocol.fail("usb_io_error:" + exc.getClass().getSimpleName()); }
            }
            persist(ownGeneration, "ERROR", "usb_io_error:" + exc.getClass().getSimpleName());
        } finally {
            closeSerial(ownGeneration);
            if (raw != null) try { raw.close(); } catch (IOException ignored) { }
            // An early open/driver failure has no protocol; retain its ERROR report.
            if (current(ownGeneration) && ownProtocol != null) persist(ownGeneration, "", "");
            running = false;
            runOnUiThread(() -> { if (!destroyed) retryButton.setEnabled(!permissionPending); });
        }
    }

    private void parse(byte[] bytes, int count, StringBuilder line, boolean[] dropping, UsbCheckProtocol check) {
        for (int i = 0; i < count; i++) {
            int value = bytes[i] & 0xff;
            if (value == '\r') continue;
            if (value == '\n') {
                if (!dropping[0] && line.length() > 0) {
                    String complete = line.toString();
                    synchronized (protocolLock) { check.acceptLine(complete, SystemClock.elapsedRealtime()); }
                    appendLine(complete);
                }
                dropping[0] = false; line.setLength(0);
            } else if (!dropping[0]) {
                if (value < 32 || value > 126 || line.length() >= 512) {
                    dropping[0] = true; line.setLength(0);
                } else line.append((char) value);
            }
        }
    }

    private void appendLine(String line) {
        runOnUiThread(() -> {
            if (destroyed) return;
            recent.append(line).append('\n');
            if (recent.length() > 8000) recent.delete(0, recent.length()-8000);
            serialText.setText(recent.toString());
        });
    }

    private void persist(long ownGeneration, String phaseOverride, String errorOverride) {
        if (ownGeneration != generation) return;
        synchronized (reportLock) {
            try {
                Map<String,Object> values = new LinkedHashMap<>();
                UsbCheckProtocol currentProtocol = protocol;
                synchronized (protocolLock) {
                    if (currentProtocol != null) values.putAll(currentProtocol.report(SystemClock.elapsedRealtime()));
                }
                if (!phaseOverride.isEmpty()) values.put("phase", phaseOverride);
                if (!errorOverride.isEmpty()) values.put("error", errorOverride);
                values.put("run_id", runId);
                values.put("written_at", Instant.now().toString());
                values.put("probe_requested", probeRequested);
                values.put("debuggable", (getApplicationInfo().flags & ApplicationInfo.FLAG_DEBUGGABLE) != 0);
                values.put("baud", 115200); values.put("data_bits", 8); values.put("stop_bits", 1); values.put("parity", "none");
                values.put("received_bytes", receivedBytes); values.put("discarded_initial_bytes", discardedBytes);
                values.put("tx_bytes", txBytes); values.put("tx_command", txBytes == 1 ? "i" : "none");
                values.put("dtr_enabled", dtrEnabled);
                values.put("raw_log", new File(reportDirectory, "latest-serial.log").getAbsolutePath());
                if (selectedDevice != null) {
                    values.put("usb_vendor_id", selectedDevice.getVendorId());
                    values.put("usb_product_id", selectedDevice.getProductId());
                    values.put("usb_device", selectedDevice.getDeviceName());
                }
                JSONObject report = new JSONObject(values);
                File temporary = new File(reportDirectory, "latest.json.tmp");
                try (FileOutputStream output = new FileOutputStream(temporary)) {
                    output.write((report.toString(2)+"\n").getBytes(StandardCharsets.UTF_8));
                }
                Files.move(temporary.toPath(), new File(reportDirectory, "latest.json").toPath(), StandardCopyOption.REPLACE_EXISTING);
                String phase = report.optString("phase", "WAIT_DEVICE");
                show(phase + (report.optString("error").isEmpty() ? "" : " · " + report.optString("error"))
                        + "\nRX " + receivedBytes + " bytes · TX " + txBytes + " bytes"
                        + "\nI2C NO ACK 也代表 USB 往返应答成功，不代表 USB 故障。");
            } catch (Exception exc) {
                show("写入诊断报告失败：" + exc.getClass().getSimpleName());
            }
        }
    }

    private void cancel(String reason) {
        boolean wasActive = running || permissionPending;
        permissionPending = false;
        if (!wasActive) {
            closeSerial(generation);
            return; // Leaving a completed screen must preserve its diagnostic result.
        }
        boolean terminal;
        synchronized (protocolLock) {
            UsbCheckProtocol currentProtocol = protocol;
            terminal = currentProtocol != null && (currentProtocol.isComplete() || currentProtocol.isFailed());
            if (currentProtocol != null) currentProtocol.fail(reason);
        }
        persist(generation, terminal ? "" : "CANCELLED", terminal ? "" : reason);
        long previous;
        synchronized (ioLock) { previous = generation++; }
        closeSerial(previous);
        if (!terminal) show("已停止 USB 检查；串口已关闭。");
        retryButton.setEnabled(!running);
    }

    private void closeSerial(long ownGeneration) {
        UsbSerialPort port;
        UsbDeviceConnection connection;
        synchronized (ioLock) {
            if (activeGeneration != ownGeneration) return;
            port = activePort; connection = activeConnection;
            activePort = null; activeConnection = null; activeGeneration = -1;
        }
        if (port != null) try { port.close(); } catch (Exception ignored) { }
        if (connection != null) connection.close();
    }

    @Override public void onDestroy() {
        cancel("activity_destroyed");
        destroyed = true;
        try { unregisterReceiver(usbReceiver); } catch (IllegalArgumentException ignored) { }
        super.onDestroy();
    }
}
