import sys
import builtins

def _ensure_isatty_on_obj(obj):
    """Ensure obj has isatty(); if it's missing, patch the class (preferred) or the instance."""
    if obj is None:
        return
    if hasattr(obj, "isatty"):
        return

    # Patch the class so any future instance is fixed too.
    cls = obj.__class__
    if not hasattr(cls, "isatty"):
        try:
            setattr(cls, "isatty", lambda self: False)
            return
        except Exception:
            pass

    # Fallback: patch the instance.
    try:
        setattr(obj, "isatty", lambda: False)
    except Exception:
        pass


def _patch_module_for_streamtologger(mod):
    """If a module exposes StreamToLogger class, patch it to have isatty()."""
    if mod is None:
        return
    stl = getattr(mod, "StreamToLogger", None)
    if stl is None:
        return
    # stl is expected to be a class
    if hasattr(stl, "__mro__") and not hasattr(stl, "isatty"):
        try:
            setattr(stl, "isatty", lambda self: False)
        except Exception:
            pass


# 1) Patch current stdout/stderr immediately (best-effort)
_ensure_isatty_on_obj(sys.stdout)
_ensure_isatty_on_obj(sys.stderr)

# 2) Install an import hook to patch StreamToLogger whenever it shows up later
_real_import = builtins.__import__

def _import_hook(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _real_import(name, globals, locals, fromlist, level)

    # Patch the top-level module and any fromlist submodules/attrs
    try:
        _patch_module_for_streamtologger(mod)
    except Exception:
        pass

    if fromlist:
        for item in fromlist:
            try:
                sub = getattr(mod, item, None)
                _patch_module_for_streamtologger(sub)
            except Exception:
                pass

    # Also: in case importing caused stdout/stderr replacement, patch again
    _ensure_isatty_on_obj(sys.stdout)
    _ensure_isatty_on_obj(sys.stderr)

    return mod

builtins.__import__ = _import_hook

# 3) Last-chance: if stdout itself is a StreamToLogger instance, patch its class too
_ensure_isatty_on_obj(sys.stdout)
_ensure_isatty_on_obj(sys.stderr)