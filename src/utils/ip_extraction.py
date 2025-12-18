"""
Unified IP extraction with trusted-proxy policy.

Provides consistent, secure client IP extraction across all middleware.
Prevents IP spoofing by validating against trusted proxy list.

IMPORTANT: Always use get_client_ip() from this module. Do NOT implement
custom IP extraction logic in middleware.
"""

import ipaddress
import logging
from typing import List, Optional

from fastapi import Request

from src.config import get_settings

logger = logging.getLogger(__name__)


def get_client_ip(request: Request) -> str:
    """
    Get client IP address with trusted-proxy validation.

    Securely extracts the client IP by:
    1. Checking X-Forwarded-For and walking back through the chain
    2. Only accepting IPs from trusted proxies
    3. Falling back to X-Real-IP, then direct connection

    SECURITY: This function validates proxy headers against the trusted
    proxy list to prevent IP spoofing attacks where a malicious client
    sends fake X-Forwarded-For headers.

    Args:
        request: The incoming FastAPI request

    Returns:
        Client IP address as string
    """
    settings = get_settings()
    trusted_proxies = settings.get_trusted_proxies_list()

    # Check X-Forwarded-For header (set by proxies/load balancers)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For can contain multiple IPs: client, proxy1, proxy2
        # Parse the chain and find the first non-trusted IP
        ips = [ip.strip() for ip in forwarded_for.split(",")]

        # If we have trusted proxies configured, walk back through the chain
        if trusted_proxies:
            for ip in reversed(ips):
                if not _is_trusted_proxy(ip, trusted_proxies):
                    return ip
            # All IPs are trusted, return the first one (original client)
            return ips[0]
        else:
            # No trusted proxies configured - log warning and return first IP
            # This is potentially unsafe in production behind a proxy
            if len(ips) > 1:
                logger.warning(
                    "X-Forwarded-For contains multiple IPs but no trusted proxies configured. "
                    "Configure TRUSTED_PROXIES to prevent IP spoofing.",
                    extra={"x_forwarded_for": forwarded_for}
                )
            return ips[0]

    # Check X-Real-IP header (set by nginx)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # Fall back to direct client IP
    if request.client:
        return request.client.host

    return "unknown"


def _is_trusted_proxy(ip: str, trusted_proxies: List[str]) -> bool:
    """
    Check if an IP is in the trusted proxy list.

    Supports both single IPs and CIDR notation (e.g., "10.0.0.0/8").

    Args:
        ip: IP address to check
        trusted_proxies: List of trusted proxy IPs or CIDRs

    Returns:
        True if IP is trusted, False otherwise
    """
    try:
        check_ip = ipaddress.ip_address(ip)

        for proxy in trusted_proxies:
            try:
                # Check if it's a CIDR range
                if "/" in proxy:
                    network = ipaddress.ip_network(proxy, strict=False)
                    if check_ip in network:
                        return True
                else:
                    # Single IP
                    if check_ip == ipaddress.ip_address(proxy):
                        return True
            except ValueError:
                continue

        return False
    except ValueError:
        return False


def validate_trusted_proxies_config() -> None:
    """
    Validate trusted proxies configuration at startup.

    Logs warnings for potentially unsafe configurations.
    Called during application startup.
    """
    settings = get_settings()
    trusted_proxies = settings.get_trusted_proxies_list()

    if not trusted_proxies:
        if settings.is_production():
            logger.warning(
                "⚠️  No trusted proxies configured in production. "
                "If running behind a load balancer, configure TRUSTED_PROXIES "
                "to prevent IP spoofing attacks."
            )
    else:
        logger.info(
            f"Trusted proxies configured: {len(trusted_proxies)} entries"
        )

        # Validate each entry
        for proxy in trusted_proxies:
            try:
                if "/" in proxy:
                    ipaddress.ip_network(proxy, strict=False)
                else:
                    ipaddress.ip_address(proxy)
            except ValueError as e:
                logger.error(
                    f"Invalid trusted proxy entry: {proxy}. Error: {e}"
                )
