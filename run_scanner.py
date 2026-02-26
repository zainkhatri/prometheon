"""Launch unzip in background."""
import subprocess
log = open("/tmp/unzip.log", "w")
subprocess.Popen(
    ["python3", "/srv/mergerfs/PROMETHEUS/PROMETHEON/do_unzip.py"],
    stdout=log, stderr=subprocess.STDOUT
)
print("Unzip launched in background. Check /tmp/unzip.log")
