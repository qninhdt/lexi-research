from setuptools import setup, find_packages

setup(
    name="lexi-research",
    version="0.1.0",
    packages=find_packages(include=["lexi_research*", "bench*", "serve*"]),
    entry_points={
        "console_scripts": [
            "lexi=lexi_research.cli:main",
        ],
    },
)
