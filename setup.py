from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")


setup(
    name="multi-llm",
    version="0.1.0",
    description="A small package to wrap multiple LLM API provider into one python SDK",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mewstake-ai/multi-llm",
    author="Julian von der Goltz",
    author_email="multi-llm@newstake.ai",
    keywords="openai, anthropic, gemini, anyscale, deepinfra, cloudflare",  # Optional
    packages=find_packages(),  # Required
    # Specify which Python versions you support. In contrast to the
    # 'Programming Language' classifiers above, 'pip install' will check this
    # and refuse to install the project if the version does not match. See
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires=">=3.7, <4",
    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/discussions/install-requires-vs-requirements/
    install_requires=[
        "openai",
        "tiktoken",
        "anthropic[vertex]"
    ],
    extras_require={},
)
