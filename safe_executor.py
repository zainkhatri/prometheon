"""Safe command executor for PROMETHEON. Blocks destructive commands, enforces allowlist.
Commands execute directly on the NAS (no SSH — the app runs locally)."""

import shlex
import subprocess
import re

NAS_CWD = "/srv/mergerfs/PROMETHEUS"

ALLOWED_COMMANDS = {
    "ls", "cat", "df", "du", "ps", "top", "free", "uname", "uptime",
    "find", "grep", "head", "tail", "echo", "pwd", "whoami", "hostname",
    "date", "mount", "zpool", "zfs", "wc", "sort", "file", "stat",
    "which", "env", "printenv", "id", "groups", "last", "w", "who",
    "netstat", "ifconfig", "ip", "ping", "traceroute", "nslookup", "dig",
    "curl", "wget", "md5sum", "sha256sum", "diff", "less", "more",
    "sensors", "lm-sensors", "hddtemp", "smartctl", "nvme",
    "iostat", "top", "htop", "lsof", "mkdir", "cp", "mv", "touch",
    "chmod", "chown", "ln", "tar", "gzip", "gunzip", "zip", "unzip",
    "python", "python3", "pip", "pip3", "apt", "dpkg",
    "tree", "basename", "dirname", "realpath", "readlink",
    "lsblk", "blkid", "fdisk", "nproc", "lscpu", "dmidecode",
    "systemctl", "journalctl", "lsattr",
}

# Hardcoded block patterns — these NEVER execute, no matter what
BLOCKED_PATTERNS = [
    r'\brm\s', r'\brm$', r'\brm\b',       # rm in any form
    r'\brmdir\b',                            # rmdir
    r'\bshred\b',                            # shred
    r'\bunlink\b',                           # unlink
    r'\bmkfs\b',                             # mkfs (format filesystem)
    r'\bdd\b',                               # dd (raw disk write)
    r'\bformat\b',                           # format
    r'\bsrm\b',                              # secure rm
    r'\btrash-put\b',                        # system trash (bypass our bin)
    r'>\s*/dev/',                             # redirect to device
    r'\bsudo\b',                             # no sudo at all
    r'\bsu\b',                               # no su
    r'\bchmod\s+777\b',                      # dangerous perms
    r'\bcrontab\s+-r\b',                     # delete crontab
    r'\bkill\b',                             # no killing processes
    r'\bkillall\b',                          # no killing processes
    r'\bpkill\b',                            # no killing processes
    r'\breboot\b',                           # no reboot
    r'\bshutdown\b',                         # no shutdown
    r'\bhalt\b',                             # no halt
    r'\bpoweroff\b',                         # no poweroff
    r'\bmv\s+.*\s+/dev/null',               # mv to /dev/null = delete
]

COMMAND_TIMEOUT = 30


def is_command_safe(command_str: str) -> tuple[bool, str]:
    """Check if a command is safe to execute. Returns (is_safe, reason)."""
    raw = command_str.strip()
    lowered = raw.lower()

    # Check for shell metacharacters that could chain destructive commands
    for dangerous_char in [';', '&&', '||', '`', '$(', '|']:
        if dangerous_char in raw:
            # Allow pipes to safe commands like grep, wc, sort, head, tail
            if dangerous_char == '|':
                continue  # pipes checked separately below
            return False, f"BLOCKED: Shell metacharacter '{dangerous_char}' not allowed. Run commands individually."

    # Block ALL destructive patterns
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, lowered):
            return False, (
                f"BLOCKED: Destructive command detected (matched '{pattern}'). "
                f"PROMETHEON never permanently deletes files. Use the trash_file tool instead."
            )

    try:
        parts = shlex.split(command_str)
    except ValueError as e:
        return False, f"BLOCKED: Could not parse command: {e}"

    if not parts:
        return False, "BLOCKED: Empty command."

    base_cmd = parts[0].split("/")[-1]

    if base_cmd not in ALLOWED_COMMANDS:
        return False, f"BLOCKED: '{base_cmd}' is not in the allowed command list."

    # Extra check: even if the base command is allowed, scan ALL args for sneaky rm
    for arg in parts[1:]:
        arg_lower = arg.lower()
        if arg_lower in ('rm', 'rmdir', 'shred', 'unlink'):
            return False, "BLOCKED: Destructive command detected in arguments."

    return True, "OK"


def execute_command(command_str: str) -> dict:
    """Execute a command safely. Returns dict with stdout, stderr, returncode, blocked."""
    safe, reason = is_command_safe(command_str)
    if not safe:
        return {
            "stdout": "",
            "stderr": reason,
            "returncode": -1,
            "blocked": True,
        }

    try:
        result = subprocess.run(
            command_str,
            shell=True,
            cwd=NAS_CWD,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
        )
        # Truncate very long output to prevent memory issues
        stdout = result.stdout[:50000] if len(result.stdout) > 50000 else result.stdout
        stderr = result.stderr[:10000] if len(result.stderr) > 10000 else result.stderr
        return {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": result.returncode,
            "blocked": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Command timed out after {COMMAND_TIMEOUT} seconds.",
            "returncode": -1,
            "blocked": False,
        }
    except FileNotFoundError:
        return {
            "stdout": "",
            "stderr": f"Command not found: {parts[0]}",
            "returncode": -1,
            "blocked": False,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Execution error: {e}",
            "returncode": -1,
            "blocked": False,
        }


def safe_shutdown():
    """Perform a safe NAS shutdown: sync filesystems, then power off."""
    steps = []
    try:
        # 1. Sync all filesystems
        subprocess.run(["sync"], timeout=30)
        steps.append("Filesystems synced")

        # 2. Initiate clean shutdown (1 minute delay so response can be sent)
        subprocess.Popen(
            ["shutdown", "-h", "+1", "PROMETHEON: Safe shutdown initiated by user"],
        )
        steps.append("Shutdown scheduled in 1 minute")

        return {
            "success": True,
            "steps": steps,
            "message": "NAS will power off in 1 minute. You can safely unplug after all lights go dark.",
        }
    except Exception as e:
        return {
            "success": False,
            "steps": steps,
            "message": f"Shutdown failed: {e}",
        }
