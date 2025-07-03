.PHONY: setup test lint clean

setup:
chmod +x setup.sh
sudo ./setup.sh

test:
pytest

lint:
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

clean:
rm -rf __pycache__
rm -rf .pytest_cache
rm -rf *.egg-info
rm -rf dist
rm -rf build
