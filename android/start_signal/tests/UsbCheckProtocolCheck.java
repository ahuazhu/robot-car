package cn.robotcar.startsignal;

/** Native JVM checks; no Android classes, USB hardware, or serial I/O. */
public final class UsbCheckProtocolCheck {
    private static int checks;
    private static final String ZERO = "DRIVE armed=0 left=0 right=0 fault=0 brake=1 lampTest=0 rxBytes=12 replies=3";
    private static final String ACK = "ESP32 I2C 0x52: ACK (presence only; not lane data)";
    private static final String NACK = "ESP32 I2C 0x52: NO ACK error=2 timeout=0";

    private static void check(boolean ok, String message) {
        ++checks;
        if (!ok) throw new AssertionError(message);
    }
    private static UsbCheckProtocol baseline(boolean probe) {
        UsbCheckProtocol protocol = new UsbCheckProtocol(probe, 0);
        protocol.acceptLine("LANE DRIVE READY | DISARMED", 10);
        protocol.acceptLine(ZERO, 1000);
        protocol.acceptLine(ZERO, 2000);
        return protocol;
    }
    private static void failed(UsbCheckProtocol protocol, String reason) {
        check(protocol.isFailed(), "failure expected: " + reason);
        check(reason.equals(protocol.report(9999).get("error")), "error reason: " + reason);
        check(!protocol.isComplete() && !protocol.shouldSendProbe(9999), "failure must prohibit probe/success");
    }
    public static void main(String[] args) {
        UsbCheckProtocol readonly = baseline(false);
        check(readonly.isComplete(), "read-only succeeds without transmitting");
        check("not_tested".equals(readonly.report(2000).get("bidirectional")), "read-only cannot claim writes");
        check(!readonly.shouldSendProbe(2000), "read-only never probes");

        for (String reply : new String[] { ACK, NACK, "ESP32 I2C 0x52: NO ACK error=5 timeout=1" }) {
            UsbCheckProtocol probe = baseline(true);
            check(probe.shouldSendProbe(2001), "baseline opens one probe gate");
            check(!probe.isComplete(), "baseline alone is not bidirectional proof");
            probe.markProbeSent(2001);
            check(!probe.shouldSendProbe(2002), "no duplicate probe");
            probe.acceptLine(ZERO, 2500);
            probe.acceptLine(ZERO, 3000);
            check(!probe.isComplete(), "pre-reply status cannot count as postflight");
            probe.acceptLine(reply + "\r", 3100);
            check("READING_POSTFLIGHT".equals(probe.phase()), "ACK and NO ACK prove USB reception");
            probe.acceptLine(ZERO, 3500);
            check(!probe.isComplete(), "one postflight status is insufficient");
            probe.acceptLine(ZERO, 4500);
            check(probe.isComplete(), "two fresh zero status lines finish");
            check("passed".equals(probe.report(4500).get("bidirectional")), "write proof recorded");
        }

        UsbCheckProtocol cached = new UsbCheckProtocol(true, 0);
        cached.acceptLine(ACK, 100);
        cached.acceptLine(ZERO, 1000);
        cached.acceptLine(NACK, 1100);
        cached.acceptLine(ZERO, 2000);
        cached.markProbeSent(2001);
        cached.acceptLine(ZERO, 3000);
        cached.acceptLine(ZERO, 4000);
        check(!cached.isComplete(), "old replies must not establish bidirectional proof");
        check("pending".equals(cached.report(4000).get("bidirectional")), "cached reply ignored");
        check(Integer.valueOf(2).equals(cached.report(4000).get("ignored_probe_replies")), "ignored replies counted");

        UsbCheckProtocol batched = new UsbCheckProtocol(true, 0);
        batched.acceptLine(ZERO, 1000);
        batched.acceptLine(ZERO, 1000);
        batched.acceptLine(ZERO, 1499);
        check(!batched.shouldSendProbe(1499), "cached batch cannot satisfy timing");
        batched.acceptLine(ZERO, 1500);
        check(batched.shouldSendProbe(1500), "500ms minimum sample gap");
        check(!batched.shouldSendProbe(3001), "last status expires after 1500ms");
        batched.acceptLine(ZERO, 4001);
        check(!batched.shouldSendProbe(4001), "stale baseline needs a fresh pair");
        batched.acceptLine(ZERO, 5001);
        check(batched.shouldSendProbe(5001), "fresh pair restores gate");

        for (String unsafe : new String[] {
                ZERO.replace("armed=0", "armed=1"), ZERO.replace("left=0", "left=1"),
                ZERO.replace("right=0", "right=-1"), ZERO.replace("fault=0", "fault=4") }) {
            UsbCheckProtocol before = new UsbCheckProtocol(true, 0);
            before.acceptLine(unsafe, 1000);
            failed(before, "unsafe_drive_state");
            UsbCheckProtocol waiting = baseline(true);
            waiting.markProbeSent(2001);
            waiting.acceptLine(unsafe, 2100);
            failed(waiting, "unsafe_drive_state");
            UsbCheckProtocol after = baseline(true);
            after.markProbeSent(2001);
            after.acceptLine(ACK, 2100);
            after.acceptLine(unsafe, 2200);
            failed(after, "unsafe_drive_state");
        }

        for (String malformed : new String[] {
                ZERO + " garbage", "DRIVE armed=0 left=0 right=0 fault=0",
                ZERO.replace("left=0", "left=9999999999999999999999999999999"),
                ZERO.replace("rxBytes=12", "rxBytes=4294967296"), ZERO.replace("fault=0", "fault=256") }) {
            UsbCheckProtocol bad = new UsbCheckProtocol(true, 0);
            bad.acceptLine(malformed, 1000);
            failed(bad, "invalid_drive_line");
        }

        for (String malformed : new String[] {
                ACK + " trailing", "ESP32 I2C 0x52: NO ACK error=0 timeout=0",
                "ESP32 I2C 0x52: NO ACK error=7 timeout=0" }) {
            UsbCheckProtocol bad = baseline(true);
            bad.markProbeSent(2001);
            bad.acceptLine(malformed, 2100);
            failed(bad, "invalid_probe_reply");
        }

        UsbCheckProtocol refused = baseline(true);
        refused.markProbeSent(2001);
        refused.acceptLine("I2C probe refused: stop first", 2100);
        failed(refused, "probe_refused_armed");
        UsbCheckProtocol restarted = baseline(true);
        restarted.markProbeSent(2001);
        restarted.acceptLine("LANE DRIVE READY | DISARMED | A0=RX A1=TX", 2100);
        failed(restarted, "uno_restarted");
        UsbCheckProtocol tooEarly = new UsbCheckProtocol(true, 0);
        tooEarly.markProbeSent(10);
        failed(tooEarly, "probe_not_ready");
        UsbCheckProtocol duplicate = baseline(true);
        duplicate.markProbeSent(2001);
        duplicate.markProbeSent(2002);
        failed(duplicate, "probe_not_ready");
        UsbCheckProtocol backwards = baseline(true);
        backwards.acceptLine(ZERO, 1999);
        failed(backwards, "non_monotonic_time");
        UsbCheckProtocol timeout = baseline(true);
        timeout.markProbeSent(2001);
        timeout.fail("deadline_exceeded");
        failed(timeout, "deadline_exceeded");
        check("failed".equals(timeout.report(14000).get("bidirectional")), "failed probe cannot claim pass");
        timeout.acceptLine(ACK, 15000);
        failed(timeout, "deadline_exceeded");

        UsbCheckProtocol postGap = baseline(true);
        postGap.markProbeSent(2001);
        postGap.acceptLine(ACK, 2100);
        postGap.acceptLine(ZERO, 3000);
        postGap.acceptLine(ZERO, 5000);
        check(!postGap.isComplete(), "postflight expired pair must reset");
        postGap.acceptLine(ZERO, 6000);
        check(postGap.isComplete(), "new postflight pair succeeds");
        System.out.println("UsbCheckProtocolCheck: " + checks + " checks passed");
    }
}
