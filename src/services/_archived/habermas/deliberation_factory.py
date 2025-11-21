"""
Factory for creating DeliberationOrchestrator with LLM support.

Handles conditional LLM integration based on configuration.
"""

import logging
from typing import Optional

from src.config.llm_config import get_llm_config
from src.services.common_ground_finder import CommonGroundFinder
from src.services.consensus_generator import ConsensusGenerator
from src.services.deliberation_orchestrator import DeliberationOrchestrator
from src.services.llm_client import LLMClient
from src.services.value_extractor import ValueExtractor

logger = logging.getLogger(__name__)

# Try to import LLM services
try:
    from src.services.consensus_generator_llm import ConsensusGeneratorLLM
    from src.services.value_extractor_llm import ValueExtractorLLM

    LLM_SERVICES_AVAILABLE = True
except ImportError:
    LLM_SERVICES_AVAILABLE = False
    logger.warning("LLM services not available")


def create_deliberation_orchestrator(
    use_llm: bool = True,
) -> DeliberationOrchestrator:
    """
    Create DeliberationOrchestrator with optional LLM support.

    Args:
        use_llm: Whether to use LLM-powered services (if available)

    Returns:
        DeliberationOrchestrator instance
    """
    # Check if LLM should be used
    should_use_llm = use_llm and LLM_SERVICES_AVAILABLE

    if should_use_llm:
        try:
            # Get LLM config
            llm_config = get_llm_config()

            # Check if API keys are configured
            if llm_config.provider == "openai" and not llm_config.openai_api_key:
                logger.warning("OpenAI API key not configured, using rule-based services")
                should_use_llm = False
            elif llm_config.provider == "anthropic" and not llm_config.anthropic_api_key:
                logger.warning(
                    "Anthropic API key not configured, using rule-based services"
                )
                should_use_llm = False

            if should_use_llm:
                # Create LLM client
                llm_client = LLMClient(config=llm_config)

                # Create LLM-powered services
                value_extractor = ValueExtractorLLM(llm_client=llm_client)
                consensus_generator = ConsensusGeneratorLLM(llm_client=llm_client)

                logger.info(
                    "Created DeliberationOrchestrator with LLM support",
                    extra={"provider": llm_config.provider},
                )

                return DeliberationOrchestrator(
                    value_extractor=value_extractor,
                    common_ground_finder=CommonGroundFinder(),
                    consensus_generator=consensus_generator,
                )

        except Exception as e:
            logger.error(f"Failed to initialize LLM services: {e}")
            logger.warning("Falling back to rule-based services")

    # Fall back to rule-based services
    logger.info("Created DeliberationOrchestrator with rule-based services")

    return DeliberationOrchestrator(
        value_extractor=ValueExtractor(),
        common_ground_finder=CommonGroundFinder(),
        consensus_generator=ConsensusGenerator(),
    )
