"""Smoke tests to verify the environment is set up correctly."""

import sys


def test_python_version():
    assert sys.version_info >= (3, 10)


def test_core_imports():
    import flask
    import numpy
    import requests
    import yaml


def test_dev_imports():
    import pytest
    import ruff
