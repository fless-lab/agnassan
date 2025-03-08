#!/usr/bin/env python
"""
Entry point for Agnassan CLI.

This script provides a simple way to start the Agnassan CLI.
"""

import asyncio
from agnassan.cli import main

if __name__ == "__main__":
    asyncio.run(main())