"""
Generate OpenAPI schema from FastAPI app.

Used for TypeScript client generation and API documentation.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app


def main():
    """Generate OpenAPI schema."""
    # Generate OpenAPI schema
    schema = app.openapi()

    # Write to file
    output_path = Path(__file__).parent.parent / "openapi.json"
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)

    print(f"✓ OpenAPI schema generated: {output_path}")
    print(f"  Title: {schema.get('info', {}).get('title')}")
    print(f"  Version: {schema.get('info', {}).get('version')}")
    print(f"  Paths: {len(schema.get('paths', {}))}")


if __name__ == "__main__":
    main()
