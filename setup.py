from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="doba",
    version="0.1.0",
    author="DoBA Team",
    author_email="doba@example.com",
    description="Document-Based AI processing system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/DoBAv2",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0,<2.0",
        "requests>=2.25.0",
        "beautifulsoup4>=4.9.0",
        "pytesseract>=0.3.8",
        "Pillow>=8.0.0",
        "duckduckgo_search>=2.0.0",
        "SpeechRecognition>=3.8.1",
    ],
)
