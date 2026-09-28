import http.server, urllib.request, sys, itertools
UP = "http://127.0.0.1:8102"; OUT = sys.argv[1]; C = itertools.count()
class H(http.server.BaseHTTPRequestHandler):
    def _fwd(self, body=None):
        req = urllib.request.Request(UP + self.path, data=body, method=self.command,
                                     headers={k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data, code, hdrs = r.read(), r.status, r.headers
        except urllib.error.HTTPError as e:
            data, code, hdrs = e.read(), e.code, e.headers
        self.send_response(code)
        for k, v in hdrs.items():
            if k.lower() not in ("transfer-encoding", "content-length", "connection"): self.send_header(k, v)
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self): self._fwd()
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        open(f"{OUT}/isl-req-{next(C)}-{self.path.strip('/').replace('/','_')}.json", "wb").write(body)
        self._fwd(body)
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("127.0.0.1", 8103), H).serve_forever()
