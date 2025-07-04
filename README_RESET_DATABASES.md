# Database Reset Tool

This document explains how to use the database reset tool for the DoBA project.

## Overview

The `reset_databases.py` script is used to safely reset all databases used by the DoBA application. It:
1. Creates a backup of all existing databases
2. Removes the existing databases
3. Creates fresh databases with the correct schema

## Running the Reset Tool

There are two ways to run the reset tool:

### 1. Using the Shell Script (Recommended)

The easiest way to run the reset tool is to use the provided shell script:

```bash
./reset_databases.sh
```

This script:
- Activates the correct virtual environment (`.venv`)
- Runs the reset_databases.py script with the proper Python interpreter

### 2. Manually with Virtual Environment

If you prefer to run the script manually, make sure to activate the virtual environment first:

```bash
source .venv/bin/activate
python reset_databases.py
```

## Troubleshooting

If you encounter an error like:
```
Cannot run program "/home/chris/DoBAv2/.venv1/bin/python" (in directory "/home/chris/DoBAv2"): error=2, No such file or directory
```

This indicates that the script is trying to use a Python interpreter from a non-existent virtual environment. Use the provided shell script instead, which ensures the correct virtual environment is used.

## Notes

- The script will ask for confirmation before resetting the databases
- All existing data will be backed up to a timestamped directory before being removed
- The backup directory will be printed at the end of the process