"""
Generate OpenAPI schema from FastAPI app.

Used for TypeScript client generation and API documentation.

Requirements:
    Python 3.11+ (as specified in pyproject.toml)
    Dependencies installed via 'poetry install'

Usage:
    poetry run python scripts/generate_openapi.py              # Generate to openapi.json
    poetry run python scripts/generate_openapi.py --output /tmp/spec.json  # Custom path
    poetry run python scripts/generate_openapi.py --check      # Check if openapi.json is current (CI)
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app


def generate_schema() -> dict:
    """Generate OpenAPI schema from FastAPI app."""
    return app.openapi()


def main():
    """Generate or validate OpenAPI schema."""
    parser = argparse.ArgumentParser(
        description="Generate OpenAPI schema from FastAPI app"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path (default: openapi.json in project root)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if committed openapi.json is current (for CI)"
    )
    args = parser.parse_args()

    # Generate OpenAPI schema
    schema = generate_schema()

    # Determine output path
    project_root = Path(__file__).parent.parent
    default_output = project_root / "openapi.json"
    output_path = Path(args.output) if args.output else default_output

    if args.check:
        # CI mode: compare generated schema with committed file
        if not default_output.exists():
            print("ERROR: openapi.json does not exist")
            sys.exit(1)

        with open(default_output, "r") as f:
            committed_schema = json.load(f)

        # Compare schemas (as formatted JSON strings for consistent comparison)
        generated_json = json.dumps(schema, indent=2, sort_keys=True)
        committed_json = json.dumps(committed_schema, indent=2, sort_keys=True)

        if generated_json != committed_json:
            print("ERROR: openapi.json is out of date!")
            print()
            print("The committed openapi.json does not match the current API.")
            print("Run 'python scripts/generate_openapi.py' to regenerate it.")
            print()

            # Find first difference
            committed_lines = committed_json.splitlines()
            generated_lines = generated_json.splitlines()
            for i, (c, g) in enumerate(zip(committed_lines, generated_lines)):
                if c != g:
                    print(f"First difference at line {i+1}:")
                    print(f"  Committed: {c[:100]}")
                    print(f"  Generated: {g[:100]}")
                    break
            else:
                if len(committed_lines) != len(generated_lines):
                    print(f"Line count differs: committed={len(committed_lines)}, generated={len(generated_lines)}")
                else:
                    print("Files appear identical but comparison failed - possible encoding issue")
            print()

            # Show summary of differences
            generated_paths = set(schema.get('paths', {}).keys())
            committed_paths = set(committed_schema.get('paths', {}).keys())

            added = generated_paths - committed_paths
            removed = committed_paths - generated_paths

            if added:
                print(f"New endpoints ({len(added)}):")
                for path in sorted(added)[:10]:
                    print(f"  + {path}")
                if len(added) > 10:
                    print(f"  ... and {len(added) - 10} more")
                print()

            if removed:
                print(f"Removed endpoints ({len(removed)}):")
                for path in sorted(removed)[:10]:
                    print(f"  - {path}")
                if len(removed) > 10:
                    print(f"  ... and {len(removed) - 10} more")
                print()

            print(f"Generated: {len(generated_paths)} paths")
            print(f"Committed: {len(committed_paths)} paths")

            sys.exit(1)

        print("OK: openapi.json is up to date")
        print(f"  Paths: {len(schema.get('paths', {}))}")
        sys.exit(0)

    # Normal mode: write schema to file with sorted keys for determinism
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2, sort_keys=True)
        f.write("\n")  # Ensure file ends with newline

    print(f"OpenAPI schema generated: {output_path}")
    print(f"  Title: {schema.get('info', {}).get('title')}")
    print(f"  Version: {schema.get('info', {}).get('version')}")
    print(f"  Paths: {len(schema.get('paths', {}))}")


if __name__ == "__main__":
    main()
