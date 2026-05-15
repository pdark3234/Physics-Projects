"""
Modified Gravity Studio — Entry Point

Launches the Flask server.
"""

import sys
import os
import builtins

os.environ.setdefault('MGS_VERBOSE', 'false')
os.environ.setdefault('MGS_SYMBOLIC_LOGS', 'false')

# Mock IPython display before any pytearcat imports
if not hasattr(builtins, 'display'):
    builtins.display = lambda *args, **kwargs: None

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and configure core mocks first
import core  # noqa: triggers __init__.py mocks

from api.app import app


def main():
    print("=" * 60)
    print("  Modified Gravity Studio — ")
    print("  http://localhost:5000")
    print("=" * 60)

    # Open browser after short delay
    import threading
    import webbrowser
    import time

    def _open_browser():
        time.sleep(2.5)
        webbrowser.open('http://localhost:5000')

    threading.Thread(target=_open_browser, daemon=True).start()

    # Run with Waitress for better thread management
    try:
        from waitress import serve
        print("Using Waitress server for improved thread management...")
        serve(app, host='0.0.0.0', port=5000, threads=4)
    except ImportError:
        print("Waitress not available, falling back to Flask dev server...")
        print("Install with: pip install waitress")
        app.run(
            debug=False,
            host='0.0.0.0',
            port=5000,
            threaded=True,
            use_reloader=False,
        )


if __name__ == '__main__':
    main()
