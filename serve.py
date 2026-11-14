#!/usr/bin/env python3
"""Static dev server for the lab, with a proxy to a local inference engine.

Two jobs:

1. Serve the lab with `Cache-Control: no-store`. Without that, edited ES modules
   keep coming from Chrome's memory cache and you debug a file you are no longer
   running.
2. Proxy `/engine/*` to a TensorFold server on 127.0.0.1:8080. The engine sends no
   CORS headers, so a page on :8777 cannot call it directly; forwarding through the
   same origin avoids that without changing the engine.

    ./serve.py [port] [engine_port]      # defaults: 8777, 8080
"""
import json
import sys
import urllib.error
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ENGINE = "http://127.0.0.1:8080"
PROXY_PREFIX = "/engine/"


class NoCacheHandler(SimpleHTTPRequestHandler):
    # HTTP/1.1 so Chrome's reused sockets are not reset mid-load. SimpleHTTPRequestHandler
    # always sends Content-Length, so persistent connections are safe here.
    protocol_version = "HTTP/1.1"

    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "404" in (fmt % args):
            super().log_message(fmt, *args)

    # ---------- engine proxy ----------

    def _proxy(self, body):
        """Forward one request to the engine.

        A `text/event-stream` answer is piped through chunk by chunk: buffering it
        would hide the one thing the streaming lesson is about, and would turn a
        token-by-token reply into a single late burst.
        """
        target = ENGINE + "/" + self.path[len(PROXY_PREFIX):]
        req = urllib.request.Request(
            target, data=body, method=self.command,
            headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
        )
        try:
            r = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:            # the engine's own refusal, verbatim
            payload = e.read()
            self._send_blob(e.code, payload)
            return
        except Exception as e:                          # engine down, or still loading weights
            self._send_blob(503, json.dumps(
                {"error": {"message": f"engine unreachable: {e}", "type": "proxy"}}).encode())
            return

        with r:
            ctype = r.headers.get("Content-Type", "application/json")
            if "text/event-stream" not in ctype:
                self._send_blob(r.status, r.read(), ctype)
                return
            # streamed: no Content-Length is known, so the connection ends the body
            self.send_response(r.status)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            while True:
                chunk = r.read1(4096) if hasattr(r, "read1") else r.read(4096)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break                               # the browser cancelled the stream

    def _send_blob(self, status, payload, ctype="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.startswith(PROXY_PREFIX):
            return self._proxy(None)
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith(PROXY_PREFIX):
            self.send_error(405)
            return
        n = int(self.headers.get("Content-Length") or 0)
        self._proxy(self.rfile.read(n) if n else None)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
    if len(sys.argv) > 2:
        ENGINE = f"http://127.0.0.1:{int(sys.argv[2])}"
    handler = partial(NoCacheHandler, directory=".")
    print(f"aiforall  → http://127.0.0.1:{port}/")
    print(f"engine    → {ENGINE} (proxied at {PROXY_PREFIX})")
    ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
