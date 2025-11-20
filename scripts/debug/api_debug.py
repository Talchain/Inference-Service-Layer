#!/usr/bin/env python3
"""
Debug utilities for API endpoints.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import httpx

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8000"


async def test_endpoint(
    endpoint: str,
    method: str = "GET",
    data: Optional[Dict[str, Any]] = None,
    base_url: str = DEFAULT_BASE_URL,
) -> None:
    """
    Test a specific API endpoint.

    Args:
        endpoint: API endpoint path (e.g., "/health", "/api/v1/causal/validate")
        method: HTTP method (GET, POST, etc.)
        data: Request payload for POST/PUT requests
        base_url: Base URL of the API server
    """
    logger.info("=" * 80)
    logger.info(f"Testing API Endpoint: {method} {endpoint}")
    logger.info("=" * 80)

    url = f"{base_url}{endpoint}"

    logger.info(f"URL: {url}")
    if data:
        logger.info(f"Request body:\n{json.dumps(data, indent=2)}")
    logger.info("")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if method.upper() == "GET":
                response = await client.get(url)
            elif method.upper() == "POST":
                response = await client.post(url, json=data)
            elif method.upper() == "PUT":
                response = await client.put(url, json=data)
            elif method.upper() == "DELETE":
                response = await client.delete(url)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return

            logger.info(f"Status code: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            logger.info("")

            try:
                response_data = response.json()
                logger.info("Response body:")
                logger.info(json.dumps(response_data, indent=2))
            except json.JSONDecodeError:
                logger.info(f"Response body (text):\n{response.text}")

            if response.status_code >= 400:
                logger.warning(f"Request failed with status {response.status_code}")
            else:
                logger.info("Request succeeded!")

    except httpx.ConnectError:
        logger.error(f"Failed to connect to {base_url}")
        logger.error("Make sure the API server is running:")
        logger.error("  uvicorn src.api.main:app --reload")
    except Exception as e:
        logger.error(f"Request failed: {e}", exc_info=True)

    logger.info("=" * 80)


async def debug_api(base_url: str = DEFAULT_BASE_URL) -> None:
    """
    Run comprehensive API debugging suite.

    Args:
        base_url: Base URL of the API server
    """
    logger.info("=" * 80)
    logger.info("API Debug Suite")
    logger.info("=" * 80)

    # Test 1: Health check
    logger.info("\n1. Testing health endpoint...")
    await test_endpoint("/health", method="GET", base_url=base_url)

    # Test 2: Causal validation
    logger.info("\n2. Testing causal validation...")
    causal_data = {
        "dag": {
            "nodes": ["Z", "X", "Y"],
            "edges": [
                {"source": "Z", "target": "X"},
                {"source": "X", "target": "Y"},
                {"source": "Z", "target": "Y"},
            ],
        },
        "treatment": "X",
        "outcome": "Y",
    }
    await test_endpoint(
        "/api/v1/causal/validate",
        method="POST",
        data=causal_data,
        base_url=base_url,
    )

    # Test 3: Preference elicitation
    logger.info("\n3. Testing preference elicitation...")
    preference_data = {
        "user_id": "debug_user",
        "context": {
            "domain": "pricing",
            "variables": ["revenue", "churn"],
            "description": "Pricing optimization",
        },
        "num_queries": 2,
    }
    await test_endpoint(
        "/api/v1/preferences/elicit",
        method="POST",
        data=preference_data,
        base_url=base_url,
    )

    logger.info("\n" + "=" * 80)
    logger.info("API Debug Suite Complete")
    logger.info("=" * 80)


def main():
    """Main entry point for API debugging."""
    import argparse

    parser = argparse.ArgumentParser(description="Debug ISL API endpoints")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the API server (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--endpoint",
        help="Specific endpoint to test (e.g., /health)",
    )
    parser.add_argument(
        "--method",
        default="GET",
        help="HTTP method (default: GET)",
    )
    parser.add_argument(
        "--data",
        help="JSON request data for POST/PUT requests",
    )

    args = parser.parse_args()

    if args.endpoint:
        data = json.loads(args.data) if args.data else None
        asyncio.run(
            test_endpoint(
                args.endpoint,
                method=args.method,
                data=data,
                base_url=args.base_url,
            )
        )
    else:
        asyncio.run(debug_api(base_url=args.base_url))


if __name__ == "__main__":
    main()
