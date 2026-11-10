"""Firmware identities implementing the existing USB Wi-Fi/pairing commands."""

CURRENT_FIRMWARE = "0.4.17-strong-rgb-or"

KNOWN_FIRMWARE = frozenset({
    "0.4.3-larger-roi",
    "0.4.6-larger-roi",
    "0.4.7-lampnet-probe",
    "0.4.8-lampnet-espnn",
    "0.4.9-lamp-face",
    "0.4.12-off-on-off",
    "0.4.13-exposure-cache",
    "0.4.14-http-lamp-config",
    "0.4.15-lamp-noise",
    "0.4.16-initial-lamp-state",
    CURRENT_FIRMWARE,
})


def supported_firmware(version):
    # Do not send credentials or pairing keys to arbitrary serial firmware.
    return isinstance(version, str) and version in KNOWN_FIRMWARE
