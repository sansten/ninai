#!/usr/bin/env python3
"""Test uvicorn server startup."""

import asyncio
import sys
from app.main import app

async def test_uvicorn_startup():
    """Test uvicorn startup."""
    try:
        print("Testing uvicorn server startup...")
        print("  1. App instance:", type(app).__name__)
        print("  2. Routes:", len(app.routes))
        print("  3. Ready to start server...")
        
        # Try to run uvicorn programmatically
        import uvicorn
        
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=8001,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        print("  4. Server config created")
        print("\n  Starting server on http://127.0.0.1:8001")
        print("  Press Ctrl+C to stop\n")
        
        await server.serve()
        
    except KeyboardInterrupt:
        print("\n✓ Server stopped by user")
        return True
    except Exception as e:
        import traceback
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        asyncio.run(test_uvicorn_startup())
    except KeyboardInterrupt:
        print("\nShutdown complete")
        sys.exit(0)
