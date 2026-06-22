#!/usr/bin/env python3
"""
Launch the NHL Game Predictor web app (Flask).
Just run: python run_app.py
"""
from app import app

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='NHL Game Predictor')
    parser.add_argument('--port', type=int, default=8501, help='Port to run on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # Avoid UnicodeEncodeError on Windows terminals that default to cp1252.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("NHL Game Predictor - Starting Flask server...")
    print("=" * 60)
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Debug: {args.debug}")
    print(f"   URL: http://{args.host}:{args.port}")
    print("-" * 60)

    app.run(host=args.host, port=args.port, debug=args.debug)