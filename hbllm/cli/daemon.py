"""CLI manager for running HBLLM as a persistent OS background daemon.

Supports:
- macOS: launchd agent (`~/Library/LaunchAgents/com.hbllm.brain.plist`)
- Linux: systemd user service (`~/.config/systemd/user/hbllm.service`)
- Direct execution: foreground runner
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


def get_log_dir() -> Path:
    log_dir = Path.home() / ".hbllm" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def install_service(args: argparse.Namespace) -> None:
    """Install and enable the background daemon service."""
    os_name = platform.system()
    python_exe = sys.executable
    work_dir = os.getcwd()
    log_dir = get_log_dir()
    log_file = log_dir / "brain.log"

    print(f"🧠 Installing HBLLM Cognitive Daemon service on {os_name}...")
    print(f"   Python: {python_exe}")
    print(f"   Working Dir: {work_dir}")
    print(f"   Log: {log_file}")

    if os_name == "Darwin":
        plist_path = Path.home() / "Library" / "LaunchAgents" / "com.hbllm.brain.plist"
        plist_path.parent.mkdir(parents=True, exist_ok=True)

        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.hbllm.brain</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_exe}</string>
        <string>-m</string>
        <string>hbllm.serving.daemon</string>
        <string>--mode</string>
        <string>systemd</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_file}</string>
    <key>StandardErrorPath</key>
    <string>{log_file}</string>
    <key>WorkingDirectory</key>
    <string>{work_dir}</string>
</dict>
</plist>
"""
        plist_path.write_text(plist_content.strip())
        print(f"✅ Created launchd agent at: {plist_path}")
        try:
            subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
            res = subprocess.run(
                ["launchctl", "load", str(plist_path)], capture_output=True, text=True
            )
            if res.returncode == 0:
                print("🚀 Successfully loaded and started service via launchctl.")
            else:
                print(f"⚠️ launchctl load note: {res.stderr.strip()}")
        except Exception as e:
            print(f"⚠️ Could not invoke launchctl directly: {e}")

    elif os_name == "Linux":
        service_path = Path.home() / ".config" / "systemd" / "user" / "hbllm.service"
        service_path.parent.mkdir(parents=True, exist_ok=True)

        service_content = f"""[Unit]
Description=HBLLM Cognitive Brain Daemon
After=network.target

[Service]
Type=simple
ExecStart={python_exe} -m hbllm.serving.daemon --mode systemd
WorkingDirectory={work_dir}
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
"""
        service_path.write_text(service_content.strip())
        print(f"✅ Created systemd service at: {service_path}")
        try:
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
            subprocess.run(["systemctl", "--user", "enable", "--now", "hbllm.service"], check=True)
            print("🚀 Successfully enabled and started hbllm.service via systemctl.")
        except Exception as e:
            print(f"⚠️ Could not enable systemd service automatically: {e}")
            print(
                "Run manually: systemctl --user daemon-reload && systemctl --user enable --now hbllm.service"
            )
    else:
        print(f"Platform '{os_name}' not yet supported for automatic service installer.")


def uninstall_service(args: argparse.Namespace) -> None:
    """Uninstall the background daemon service."""
    os_name = platform.system()

    if os_name == "Darwin":
        plist_path = Path.home() / "Library" / "LaunchAgents" / "com.hbllm.brain.plist"
        if plist_path.exists():
            subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
            plist_path.unlink()
            print("✅ Unloaded and removed launchd service: com.hbllm.brain")
        else:
            print("Service not found at", plist_path)
    elif os_name == "Linux":
        service_path = Path.home() / ".config" / "systemd" / "user" / "hbllm.service"
        if service_path.exists():
            subprocess.run(["systemctl", "--user", "stop", "hbllm.service"], capture_output=True)
            subprocess.run(["systemctl", "--user", "disable", "hbllm.service"], capture_output=True)
            service_path.unlink()
            subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
            print("✅ Stopped and removed systemd service: hbllm.service")
        else:
            print("Service not found at", service_path)


def status_service(args: argparse.Namespace) -> None:
    """Check running status of background service."""
    os_name = platform.system()
    print(f"🔍 Checking HBLLM Daemon status ({os_name})...")

    if os_name == "Darwin":
        res = subprocess.run(["launchctl", "list"], capture_output=True, text=True)
        if "com.hbllm.brain" in res.stdout:
            for line in res.stdout.splitlines():
                if "com.hbllm.brain" in line:
                    print("🟢 Service is REGISTERED and RUNNING:")
                    print(f"   {line}")
        else:
            print("🔴 Service 'com.hbllm.brain' is NOT currently active.")
    elif os_name == "Linux":
        res = subprocess.run(
            ["systemctl", "--user", "status", "hbllm.service"], capture_output=True, text=True
        )
        print(res.stdout or res.stderr)

    log_file = get_log_dir() / "brain.log"
    if log_file.exists():
        print(f"\n📄 Recent Logs ({log_file}):")
        try:
            lines = log_file.read_text().splitlines()
            for l in lines[-10:]:
                print(f"   {l}")
        except Exception:
            pass


def run_foreground(args: argparse.Namespace) -> None:
    """Run daemon in current terminal foreground."""
    from hbllm.serving.daemon import main as daemon_main

    sys.argv = [sys.argv[0]]
    if getattr(args, "provider", None):
        sys.argv.extend(["--provider", args.provider])
    if getattr(args, "local", False):
        sys.argv.append("--local")
    if getattr(args, "port", None):
        sys.argv.extend(["--port", str(args.port)])

    daemon_main()


def register_daemon_subcommands(subparsers: argparse._SubParsersAction) -> None:
    """Register daemon command and subcommands into the main CLI parser."""
    parser = subparsers.add_parser("daemon", help="Manage always-on cognitive daemon service")
    sub = parser.add_subparsers(dest="daemon_action", required=True)

    # run (foreground)
    run_p = sub.add_parser("run", help="Run cognitive daemon in foreground")
    run_p.add_argument("--provider", default="openai/gpt-4o-mini", help="LLM Provider")
    run_p.add_argument("--local", action="store_true", help="Run local model")
    run_p.add_argument("--port", type=int, default=8000, help="HTTP Port")

    # install
    install_p = sub.add_parser("install", help="Install persistent OS background daemon")
    install_p.add_argument("--provider", default=None, help="Default LLM provider")

    # uninstall
    sub.add_parser("uninstall", help="Uninstall background daemon service")

    # status
    sub.add_parser("status", help="Check status and logs of background daemon")
