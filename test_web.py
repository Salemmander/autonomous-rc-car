#!/usr/bin/env python3
"""Test the web controller locally."""
import sys
sys.path.insert(0, '.')

from web_controller.web import LocalWebController

print("Starting web controller...")
print("Open https://localhost:8887 in your browser")
print("(Accept the self-signed certificate warning)")
print("Press Ctrl+C to stop\n")

controller = LocalWebController()
controller.update()
