"""Record-only HTTP proxy between local PLoT and local ISL.

Every request PLoT sends to ISL, and ISL's reply, is written verbatim to the capture
directory so the runner can show the ACTUAL translated engine request. It never edits
a body. Stdlib only; listens on 127.0.0.1.
"""
import http.server, json, os, sys, threading, time, urllib.request, urllib.error

UPSTREAM = sys.argv[1]            # e.g. http://127.0.0.1:18731
PORT = int(sys.argv[2])           # e.g. 18733
OUT = sys.argv[3]                 # capture directory
os.makedirs(OUT, exist_ok=True)
lock = threading.Lock(); counter = [0]

class Proxy(http.server.BaseHTTPRequestHandler):
    def _forward(self):
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection", "accept-encoding")}
        req = urllib.request.Request(UPSTREAM + self.path, data=body, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(req, timeout=170) as r:
                status, rheaders, rbody = r.status, r.headers.items(), r.read()
        except urllib.error.HTTPError as e:
            status, rheaders, rbody = e.code, e.headers.items(), e.read()
        if self.command == "POST":
            with lock:
                counter[0] += 1; n = counter[0]
            rec = {"seq": n, "t": time.time(), "method": self.command, "path": self.path,
                   "request": json.loads(body) if body else None, "status": status}
            try: rec["response"] = json.loads(rbody)
            except Exception: rec["response_text"] = rbody.decode("utf-8", "replace")
            with open(os.path.join(OUT, f"{n:04d}.json"), "w") as f: json.dump(rec, f)
        self.send_response(status)
        for k, v in rheaders:
            if k.lower() not in ("transfer-encoding", "connection", "content-length"): self.send_header(k, v)
        self.send_header("content-length", str(len(rbody))); self.end_headers(); self.wfile.write(rbody)
    do_GET = do_POST = _forward
    def log_message(self, *a): pass

http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Proxy).serve_forever()
