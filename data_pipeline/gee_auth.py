"""
data_pipeline/gee_auth.py
=========================
Utility for managing Google Earth Engine authentication.

GEE requires a Google Cloud project with the Earth Engine API enabled.

Setup instructions:
  1. Sign up for GEE access: https://signup.earthengine.google.com
  2. Create a GCP project:   https://console.cloud.google.com
  3. Enable Earth Engine API: https://console.cloud.google.com/apis/library/earthengine.googleapis.com
  4. Authenticate (run ONCE in terminal):
       earthengine authenticate
  5. Set GEE_PROJECT_ID=your-gcp-project-id in .env
  6. Verify: python -m data_pipeline.gee_auth
"""

from __future__ import annotations
import os
from dotenv import load_dotenv


def verify_gee_credentials(project_id: str = "") -> bool:

    load_dotenv()
    pid = project_id or os.getenv("GEE_PROJECT_ID", "")

    if not pid:
        print("[FAIL] GEE_PROJECT_ID not set. Check your .env file.")
        return False

    try:
        import ee
        ee.Initialize(project=pid)
        # Quick connectivity test: fetch a known image
        test = ee.Image("USGS/SRTMGL1_003").getInfo()
        if test:
            print(f"[OK  ] GEE initialized successfully (project: {pid})")
            return True
    except ImportError:
        print("[FAIL] earthengine-api not installed. "
              "Run: pip install earthengine-api")
        return False
    except Exception as e:
        print(f"[FAIL] GEE initialization failed: {e}")
        print("       Run `earthengine authenticate` in your terminal.")
        return False
    return False


if __name__ == "__main__":
    verify_gee_credentials()