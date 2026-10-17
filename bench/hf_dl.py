import os, sys, json, urllib.request, urllib.error
repo, dest = sys.argv[1], os.path.expanduser(sys.argv[2])
os.makedirs(dest, exist_ok=True)
info = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}?blobs=true"))
files = {s["rfilename"]: s.get("size") for s in info["siblings"]}
has_st = any(n.endswith(".safetensors") for n in files)
SKIP = (".bin", ".pt", ".pth", ".h5", ".msgpack", ".onnx", ".gguf", ".ot")
bad = []
for name, size in files.items():
    if name.startswith("original/") or (has_st and name.endswith(SKIP)):
        continue
    path = os.path.join(dest, name)
    os.makedirs(os.path.dirname(path) or dest, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    for attempt in range(60):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if size is not None and have == size:
            break
        if size is not None and have > size:
            os.remove(path)
            have = 0
        hdr = {"Range": f"bytes={have}-"} if have else {}
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=60)
            if have and r.status != 206:      # server ignored Range: restart this file
                have = 0
            with r, open(path, "ab" if have else "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
        except urllib.error.HTTPError as e:
            print("retry", name, e)
            if e.code == 416 and os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print("retry", name, e)
    got = os.path.getsize(path) if os.path.exists(path) else 0
    ok = (size is None) or (got == size)
    print("OK " if ok else "BAD", name, got, size, flush=True)
    if not ok:
        bad.append(name)
if bad:
    print("INCOMPLETE:", bad)
    sys.exit(1)
print("all files complete")