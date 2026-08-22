"""tau-research: Specialized Customer-Service Agent via SFT -> Agentic RL on tau3-bench."""

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

__version__ = "0.1.0"
