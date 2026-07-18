"""Async synthetic-data generator package."""

from .teacher import BaseTeacher, DeepSeekTeacher, build_teacher
from .templates import Template, TEMPLATES, get_template, list_templates, extract_json
from .prompts import SeedProvider, Seed, render, parse_response, build_sample
from .batching import BatchGenerator

__all__ = [
    "BaseTeacher",
    "DeepSeekTeacher",
    "build_teacher",
    "Template",
    "TEMPLATES",
    "get_template",
    "list_templates",
    "extract_json",
    "SeedProvider",
    "Seed",
    "render",
    "parse_response",
    "build_sample",
    "BatchGenerator",
]
