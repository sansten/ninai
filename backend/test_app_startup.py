#!/usr/bin/env python3
"""Test app startup directly."""

import sys
import asyncio

async def test_startup():
    """Test that the app can start up."""
    print("Testing app startup...")
    try:
        print("  1. Importing create_application...")
        from app.main import create_application
        print("     ✓ Import successful")
        
        print("  2. Creating FastAPI app instance...")
        app = create_application()
        print("     ✓ App created successfully")
        
        print(f"  3. App details:")
        print(f"     - Routes: {len(app.routes)}")
        print(f"     - Title: {app.title}")
        
        return True
    except Exception as e:
        import traceback
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_startup())
    sys.exit(0 if result else 1)
