package cn.robotcar.startsignal;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * One USB-only, zero-output diagnostic session; this class never performs I/O.
 * The caller owns an exclusive serial reader, drains old input before writing
 * the single byte 'i', and calls markProbeSent only after that write succeeds.
 * All timestamps use the same monotonic millisecond clock. UNO's existing probe
 * has no request nonce, so an exclusive connection and input drain are required.
 */
public final class UsbCheckProtocol {
    private static final long MIN_SAMPLE_GAP_MS = 500;
    private static final long MAX_STATUS_AGE_MS = 1500;
    private static final Pattern DRIVE = Pattern.compile(
            "DRIVE armed=([01]) left=(-?[0-9]+) right=(-?[0-9]+) fault=([0-9]+) "
            + "brake=([01]) lampTest=([01]) rxBytes=([0-9]+) replies=([0-9]+)");
    private static final String ACK = "ESP32 I2C 0x52: ACK (presence only; not lane data)";
    private static final Pattern NO_ACK = Pattern.compile(
            "ESP32 I2C 0x52: NO ACK error=([0-5]) timeout=([01])");

    private enum Phase {
        READING_BASELINE, READY_TO_PROBE, WAIT_PROBE_REPLY,
        READING_POSTFLIGHT, COMPLETE, FAILED
    }

    private final boolean probeRequested;
    private final long openedAtMs;
    private Phase phase = Phase.READING_BASELINE;
    private long lastEventAtMs;
    private long lastZeroAtMs = -1;
    private long sampleStartedAtMs = -1;
    private long probeSentAtMs = -1;
    private int baselineSamples;
    private int postflightSamples;
    private int driveLines;
    private int ignoredProbeReplies;
    private String probeReply = "";
    private String probeResult = "not_received";
    private String error = "";

    public UsbCheckProtocol(boolean probeRequested, long openedAtMs) {
        if (openedAtMs < 0) throw new IllegalArgumentException("negative monotonic timestamp");
        this.probeRequested = probeRequested;
        this.openedAtMs = openedAtMs;
        this.lastEventAtMs = openedAtMs;
    }

    public void acceptLine(String line, long nowMs) {
        if (isComplete() || isFailed()) return;
        if (!acceptTime(nowMs)) return;
        if (line == null) return;
        if (line.endsWith("\r")) line = line.substring(0, line.length() - 1);
        if (line.startsWith("DRIVE ")) {
            acceptDrive(line, nowMs);
        } else if (line.startsWith("ESP32 I2C 0x52:")) {
            if (phase != Phase.WAIT_PROBE_REPLY) {
                ++ignoredProbeReplies;
                return;
            }
            Matcher noAck = NO_ACK.matcher(line);
            if (ACK.equals(line)) probeResult = "ack";
            else if (noAck.matches()
                    && !("0".equals(noAck.group(1)) && "0".equals(noAck.group(2)))) {
                probeResult = "no_ack";
            } else {
                fail("invalid_probe_reply");
                return;
            }
            probeReply = line;
            phase = Phase.READING_POSTFLIGHT;
            lastZeroAtMs = -1;
            sampleStartedAtMs = -1;
        } else if (phase == Phase.WAIT_PROBE_REPLY
                && "I2C probe refused: stop first".equals(line)) {
            fail("probe_refused_armed");
        } else if (driveLines > 0 && line.startsWith("LANE DRIVE READY")) {
            fail("uno_restarted");
        }
    }

    private void acceptDrive(String line, long nowMs) {
        Matcher match = DRIVE.matcher(line);
        if (!match.matches()) {
            fail("invalid_drive_line");
            return;
        }
        long armed, left, right, fault, rxBytes, replies;
        try {
            armed = Long.parseLong(match.group(1));
            left = Long.parseLong(match.group(2));
            right = Long.parseLong(match.group(3));
            fault = Long.parseLong(match.group(4));
            rxBytes = Long.parseLong(match.group(7));
            replies = Long.parseLong(match.group(8));
        } catch (NumberFormatException ex) {
            fail("invalid_drive_line");
            return;
        }
        if (fault > 255 || rxBytes > 0xffffffffL || replies > 0xffffffffL) {
            fail("invalid_drive_line");
            return;
        }
        ++driveLines;
        if (armed != 0 || left != 0 || right != 0 || fault != 0) {
            fail("unsafe_drive_state");
            return;
        }

        // Status while waiting for the reply is monitored, but cannot satisfy
        // the required two fresh status lines after that reply.
        if (phase == Phase.WAIT_PROBE_REPLY) return;
        boolean postflight = phase == Phase.READING_POSTFLIGHT;
        int samples = postflight ? postflightSamples : baselineSamples;
        if (samples == 0 || lastZeroAtMs < 0 || nowMs - lastZeroAtMs > MAX_STATUS_AGE_MS) {
            samples = 1;
            sampleStartedAtMs = nowMs;
        } else if (samples == 1 && nowMs - sampleStartedAtMs >= MIN_SAMPLE_GAP_MS) {
            samples = 2;
        }
        lastZeroAtMs = nowMs;
        if (postflight) {
            postflightSamples = samples;
            if (samples >= 2) phase = Phase.COMPLETE;
        } else {
            baselineSamples = samples;
            if (samples >= 2) phase = probeRequested ? Phase.READY_TO_PROBE : Phase.COMPLETE;
            else phase = Phase.READING_BASELINE;
        }
    }

    public boolean shouldSendProbe(long nowMs) {
        return probeRequested && phase == Phase.READY_TO_PROBE
                && nowMs >= lastEventAtMs && nowMs >= lastZeroAtMs
                && nowMs - lastZeroAtMs <= MAX_STATUS_AGE_MS;
    }

    /** Called only once, after successful write(new byte[] { 'i' }). */
    public void markProbeSent(long nowMs) {
        if (isFailed() || isComplete()) return;
        if (!shouldSendProbe(nowMs)) {
            fail("probe_not_ready");
            return;
        }
        lastEventAtMs = nowMs;
        probeSentAtMs = nowMs;
        phase = Phase.WAIT_PROBE_REPLY;
    }

    private boolean acceptTime(long nowMs) {
        if (nowMs < lastEventAtMs) {
            fail("non_monotonic_time");
            return false;
        }
        lastEventAtMs = nowMs;
        return true;
    }

    /** Serial/permission/deadline errors are supplied by the I/O owner. */
    public void fail(String reason) {
        if (isComplete() || isFailed()) return;
        error = reason == null || reason.isEmpty() ? "io_failed" : reason;
        phase = Phase.FAILED;
    }

    public boolean isComplete() { return phase == Phase.COMPLETE; }
    public boolean isFailed() { return phase == Phase.FAILED; }
    public String phase() { return phase.name(); }

    public Map<String, Object> report(long nowMs) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("mode", probeRequested ? "i2c_presence_probe" : "read_only");
        report.put("phase", phase());
        report.put("success", isComplete());
        report.put("zero_output_confirmed", isComplete());
        report.put("bidirectional", !probeReply.isEmpty() ? "passed"
                : probeSentAtMs < 0 ? "not_tested" : isFailed() ? "failed" : "pending");
        report.put("probe_sent", probeSentAtMs >= 0);
        report.put("probe_result", probeResult);
        report.put("probe_reply", probeReply);
        report.put("ignored_probe_replies", ignoredProbeReplies);
        report.put("drive_lines", driveLines);
        report.put("preflight_zero_samples", baselineSamples);
        report.put("postflight_zero_samples", postflightSamples);
        report.put("elapsed_ms", Math.max(0, nowMs - openedAtMs));
        report.put("error", error);
        return report;
    }
}
