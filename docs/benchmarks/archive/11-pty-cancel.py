import os, pty, sys, time, select, struct, fcntl, termios, re
port = sys.argv[1]
pid, fd = pty.fork()
if pid == 0:
    os.environ.update(TERM="xterm-256color", COLUMNS="100", LANG="en_US.UTF-8", NO_COLOR="1")
    os.execv("/home/mitali/miracles/marin/target/release/marin", ["marin", "code", "/home/mitali/miracles/marin", "--port", port])
fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
out = b""
def pump(secs):
    global out
    end = time.time() + secs
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.2)
        if r:
            try: d = os.read(fd, 65536)
            except OSError: return
            if not d: return
            out += d
def clean(b): return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]|\r", "", b.decode("utf8", "replace"))
pump(1.5)
os.write(fd, b"Explain how a streamed chat completion request travels through the server in detail.\r")
t0 = time.time(); pump(4.0)
os.write(fd, b"\x03"); t1 = time.time()
deadline = time.time() + 20
while time.time() < deadline and b"cancelled" not in out: pump(0.3)
print(f"Ctrl-C to 'cancelled' notice: {time.time()-t1:.2f} s")
pump(1.0)
os.write(fd, b"What is the default port of marin serve?\r")
deadline = time.time() + 90
while time.time() < deadline and out.count(b"tokens written") < 1: pump(0.5)
pump(0.5)
os.write(fd, b"\x04"); pump(1.5)
try:
    _, status = os.waitpid(pid, os.WNOHANG)
except ChildProcessError: status = None
text = clean(out)
i = text.find("Explain how")
print(text[i:][-1400:])
print("exit status:", status)
