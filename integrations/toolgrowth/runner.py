"""Sandbox child process. Reads the tool source, restricts builtins and imports, arms a PEP 578 audit hook, runs the tool on a list of inputs, writes results to a pre-opened fd.
Usage: python3 -I runner.py WORKDIR FLAGS   (FLAGS: comma list of 'builtins', 'hook')"""
import json, os, sys, sysconfig

work, flags = sys.argv[1], set(sys.argv[2].split(","))
src = open(os.path.join(work, "tool.py")).read()
cases = json.load(open(os.path.join(work, "in.json")))
out_fd = os.open(os.path.join(work, "out.json"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
ALLOWED = set(json.load(open(os.path.join(work, "allowed.json"))))
import builtins as _b
_real_import = _b.__import__

if "builtins" in flags:
    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split(".")[0] not in ALLOWED:
            raise ImportError(f"module '{name}' is not permitted in a tool")
        return _real_import(name, globals, locals, fromlist, level)
    keep = ("abs all any ascii bin bool bytearray bytes callable chr complex dict divmod enumerate filter float format frozenset hash hex int isinstance issubclass iter len list map max min next oct ord pow "
            "range repr reversed round set slice sorted str sum tuple type zip True False None Exception ValueError TypeError KeyError IndexError ZeroDivisionError ArithmeticError StopIteration "
            "OverflowError RuntimeError AssertionError NotImplementedError LookupError UnicodeError").split()
    safe = {k: getattr(_b, k) for k in keep}
    safe["__import__"] = guarded_import
    safe["print"] = lambda *a, **k: None
    env = {"__builtins__": safe, "__name__": "tool"}
else:
    env = {"__name__": "tool"}

allowed_read_roots = tuple(sorted({p for p in list(sysconfig.get_paths().values()) + [sys.prefix, sys.base_prefix, sys.exec_prefix] if p}))
DENY_PREFIXES = ("socket.", "subprocess.", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "os.kill", "os.killpg", "os.remove", "os.rename", "os.replace", "os.mkdir", "os.rmdir",
                 "os.chmod", "os.chown", "os.truncate", "os.symlink", "os.link", "os.mkfifo", "os.mknod", "os.utime", "os.chdir", "os.chroot", "os.putenv", "os.unsetenv", "ctypes.", "fcntl.", "mmap.", "pty.",
                 "shutil.", "tempfile.", "glob.", "sys.settrace", "sys.setprofile", "urllib.", "http.", "webbrowser.", "ftplib.", "smtplib.", "sqlite3.", "pickle.", "marshal.", "winreg.", "resource.")
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def violation(msg):
    os.write(2, f"VIOLATION:{msg}\n".encode()[:500]); os._exit(97)


def hook(event, args):
    if event == "open":
        path, _mode, flg = args
        if isinstance(path, int): violation(f"open fd {path}")
        p = os.fsdecode(path) if isinstance(path, (str, bytes, os.PathLike)) else str(path)
        if (flg or 0) & WRITE_FLAGS: violation(f"open for write {p}")
        if not os.path.abspath(p).startswith(allowed_read_roots): violation(f"open for read {p}")
    elif event in ("os.listdir", "os.scandir"):
        p = os.fsdecode(args[0]) if args and isinstance(args[0], (str, bytes, os.PathLike)) else "."
        if not os.path.abspath(p).startswith(allowed_read_roots): violation(f"{event} {p}")
    elif event.startswith(DENY_PREFIXES):
        violation(event)

try:
    code = compile(src, "tool.py", "exec")
    exec(code, env)
    run = env.get("run")
    if not callable(run): raise RuntimeError("no callable `run`")
    if "hook" in flags: sys.addaudithook(hook)
    results = []
    for c in cases:
        try: results.append({"ok": True, "output": run(c)})
        except SystemExit: raise
        except BaseException as e: results.append({"ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"})
    data = json.dumps(results, allow_nan=False).encode()
except SystemExit:
    data = json.dumps([{"ok": False, "error": "SystemExit"}]).encode()
except BaseException as e:
    data = json.dumps({"fatal": f"{type(e).__name__}: {str(e)[:200]}"}).encode()
os.write(out_fd, data)
os._exit(0)
