from setuptools import setup

setup(
    name="npsem",
    packages=["npsem", "npsem.NIPS2018POMIS_exp"],
    version="0.1.0",
    author="Sanghack Lee",
    author_email="sanghack.lee@gmail.com",
    description="Structural Causal Bandits: Where to Intervene?",
    python_requires=">=3.9",
    install_requires=[
        "numpy>=2.3.0",
        "scipy>=1.16.0", 
        "joblib>=1.5.2",
        "matplotlib>=3.10.0",
        "seaborn>=0.13.0",
        "networkx>=3.5.0",
        "pandas>=2.3.0",
        "ananke-causal>=0.5.0",
        "causal-learn>=0.1.4.0",
        "rpy2>=3.6.0",
        "scikit-learn>=1.7.0",
        "pyro-ppl>=1.9.0",
        "tqdm>=4.67.0",
        "click>=8.2.0",
    ],
    extras_require={
        "dev": [
            "ruff>=0.12.0",
        ],
        "tracking": [
            "wandb>=0.21.0",
        ],
    },
)
