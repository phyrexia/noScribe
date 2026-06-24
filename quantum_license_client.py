"""
Quantum License Client — Python SDK.

Drop this single file into a project and call `gate()` at the top of the
entrypoint. Without a valid token, the process exits.

Fingerprint = SHA-256(machine_id + hostname + user + primary_nic_mac).

The server tells the client its tier on activation/heartbeat, which sets the
heartbeat cadence dynamically.

Usage:
    from quantum_license_client import gate
    gate(
        license_id="bcg-aimemory-001",
        repo="ai-memory",
        server="https://quantum-license-server-653349625911.us-central1.run.app",
    )
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

CACHE_DIR = Path(os.path.expanduser("~")) / ".quantum-license"
CACHE_DIR.mkdir(mode=0o700, exist_ok=True)

DEFAULT_HEARTBEAT_S = 3600  # before first response, conservative


# --- Fingerprinting ---

def _machine_id() -> str:
    """Cross-platform stable machine identifier."""
    system = platform.system()
    if system == "Linux":
        for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            try:
                return Path(path).read_text().strip()
            except FileNotFoundError:
                continue
        return socket.gethostname()
    if system == "Darwin":
        try:
            out = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            for line in out.splitlines():
                if "IOPlatformUUID" in line:
                    return line.split('"')[-2]
        except Exception:
            pass
        return socket.gethostname()
    if system == "Windows":
        try:
            out = subprocess.run(
                ["wmic", "csproduct", "get", "UUID"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            lines = [l.strip() for l in out.splitlines() if l.strip() and "UUID" not in l]
            if lines:
                return lines[0]
        except Exception:
            pass
        return socket.gethostname()
    return socket.gethostname()


def _primary_nic_mac() -> str:
    """MAC of the primary chassis NIC. Skips USB-Ethernet, loopback, virtual.

    Linux: prefers en0/eth0/enp* in /sys/class/net.
    macOS: prefers en0 (built-in Wi-Fi or Ethernet, depending on model).
    Windows: first non-virtual physical adapter.
    """
    system = platform.system()
    try:
        if system == "Linux":
            base = Path("/sys/class/net")
            candidates = []
            for nic in base.iterdir():
                name = nic.name
                if name == "lo" or name.startswith(("docker", "veth", "br-", "tun", "tap", "vbox", "virbr")):
                    continue
                addr_file = nic / "address"
                if addr_file.exists():
                    mac = addr_file.read_text().strip()
                    if mac and mac != "00:00:00:00:00:00":
                        # Skip USB-Eth: their device path contains /usb
                        dev_path = (nic / "device").resolve() if (nic / "device").exists() else None
                        is_usb = dev_path and "/usb" in str(dev_path)
                        candidates.append((name.startswith(("en", "eth", "wl")), not is_usb, name, mac))
            if candidates:
                # Prefer en/eth/wl prefix, then non-USB, then alphabetical name
                candidates.sort(key=lambda c: (not c[0], not c[1], c[2]))
                return candidates[0][3]
        elif system == "Darwin":
            out = subprocess.run(
                ["ifconfig", "en0"], capture_output=True, text=True, timeout=5,
            ).stdout
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("ether "):
                    return line.split()[1]
        elif system == "Windows":
            out = subprocess.run(
                ["getmac", "/fo", "csv", "/nh"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            for line in out.splitlines():
                parts = [p.strip('"') for p in line.split(",")]
                if len(parts) >= 2 and "PCI" in parts[1].upper():
                    return parts[0]
    except Exception:
        pass
    return "00:00:00:00:00:00"


def machine_license_id(app: str) -> str:
    """Per-machine license id: '<app>-<user>-<device12>'.

    Derives a stable, unique id from the app name, the OS user, and a hash of
    the machine id, so each machine self-identifies instead of sharing one
    hardcoded license that gets fingerprint-locked to a single machine.
    """
    user = getpass.getuser()
    dev = hashlib.sha256(_machine_id().encode()).hexdigest()[:12]
    base = "".join(c if c.isalnum() else "-" for c in f"{app}-{user}").lower()
    return f"{base}-{dev}"


def _fingerprint() -> tuple[str, dict]:
    mid = _machine_id()
    host = socket.gethostname()
    user = getpass.getuser()
    mac = _primary_nic_mac()
    raw = f"{mid}|{host}|{user}|{mac}"
    fp = hashlib.sha256(raw.encode()).hexdigest()
    details = {
        "hostname": host,
        "user": user,
        "machine_id_prefix": mid[:8] + "..." if len(mid) > 8 else mid,
        "primary_nic_mac": mac,
        "os": platform.system(),
    }
    return fp, details


# --- HTTP helpers ---

def _post(url: str, body: dict, timeout: float = 10.0) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# --- Cache ---

def _cache_path(license_id: str) -> Path:
    return CACHE_DIR / f"{license_id}.json"


def _load_cache(license_id: str) -> dict | None:
    p = _cache_path(license_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _save_cache(license_id: str, data: dict) -> None:
    p = _cache_path(license_id)
    p.write_text(json.dumps(data))
    p.chmod(0o600)


# --- Core ---

def _activate(server: str, license_id: str) -> dict:
    fp, details = _fingerprint()
    resp = _post(f"{server}/v1/activate", {
        "license_id": license_id,
        "fingerprint": fp,
        "fingerprint_details": details,
    })
    resp["fingerprint"] = fp
    return resp


def _heartbeat(server: str, token: str, license_id: str) -> dict:
    resp = _post(f"{server}/v1/heartbeat", {"token": token})
    fp, _ = _fingerprint()
    resp["fingerprint"] = fp
    return resp


def _supervisor(server: str, license_id: str, repo: str):
    """Daemon thread: refreshes token at tier-driven cadence, exits process if it can't."""
    fail_count = 0
    while True:
        cache = _load_cache(license_id) or {}
        token = cache.get("token")
        exp = int(cache.get("expires_at", 0))
        heartbeat_s = int(cache.get("heartbeat_seconds", DEFAULT_HEARTBEAT_S))

        now = int(time.time())
        if now >= exp:
            # Token expired offline. Game over.
            sys.stderr.write(f"[quantum-license:{repo}] token expired offline. Shutting down.\n")
            os._exit(2)

        # Sleep until next heartbeat (with jitter ±10%)
        import random
        sleep_for = max(30, heartbeat_s + int(random.uniform(-0.1, 0.1) * heartbeat_s))
        time.sleep(sleep_for)

        try:
            resp = _heartbeat(server, token, license_id)
            _save_cache(license_id, resp)
            fail_count = 0
        except Exception as e:
            fail_count += 1
            sys.stderr.write(f"[quantum-license:{repo}] heartbeat failed ({fail_count}): {e}\n")
            # If exp is far enough, we tolerate failures until it actually expires.
            if int(time.time()) >= exp:
                sys.stderr.write(f"[quantum-license:{repo}] token now expired after heartbeat failures. Shutting down.\n")
                os._exit(2)


def gate(license_id: str, repo: str, server: str) -> None:
    """Block at startup if license invalid. Spawn supervisor thread on success.

    Calls sys.exit(1) if activation/initial heartbeat fails AND no valid cached
    token exists.
    """
    cache = _load_cache(license_id)
    now = int(time.time())

    if cache and int(cache.get("expires_at", 0)) > now:
        # Valid cached token. Try a non-blocking heartbeat refresh, but proceed
        # either way — supervisor thread will retry.
        try:
            resp = _heartbeat(server, cache["token"], license_id)
            _save_cache(license_id, resp)
        except Exception:
            pass  # supervisor will handle
    else:
        try:
            resp = _activate(server, license_id)
            _save_cache(license_id, resp)
        except Exception as e:
            sys.stderr.write(f"[quantum-license:{repo}] activation failed and no valid cached token: {e}\n")
            sys.stderr.write(f"[quantum-license:{repo}] License: {license_id}. Contact owner.\n")
            sys.exit(1)

    t = threading.Thread(target=_supervisor, args=(server, license_id, repo), daemon=True)
    t.start()
