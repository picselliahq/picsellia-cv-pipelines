import sys
import io

def _wrap_with_isatty(stream):
    # If the stream already has isatty, nothing to do
    if hasattr(stream, "isatty"):
        return stream

    # Try to attach isatty dynamically (works for most objects)
    try:
        stream.isatty = lambda: False
        return stream
    except Exception:
        # Fallback: wrap the stream
        class StreamWrapper(io.TextIOBase):
            def __init__(self, s):
                self._s = s
            def write(self, data):
                return self._s.write(data)
            def flush(self):
                return self._s.flush()
            def isatty(self):
                return False
            def __getattr__(self, name):
                return getattr(self._s, name)

        return StreamWrapper(stream)

sys.stdout = _wrap_with_isatty(sys.stdout)
sys.stderr = _wrap_with_isatty(sys.stderr)