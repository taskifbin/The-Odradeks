#!/usr/bin/env python3
"""Wrapper delegating to tests/test_public_samples.py"""
import os
import sys

target = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "..", "..", "tests", "test_public_samples.py")
target = os.path.abspath(target)
if not os.path.exists(target):
    target = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "..", "tests", "test_public_samples.py")
    target = os.path.abspath(target)

os.execv(sys.executable, [sys.executable, target] + sys.argv[1:])
