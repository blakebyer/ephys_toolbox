"""Command line interface for Ephys toolbox"""
import click
import os
import warnings
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, List, Tuple

def parse_multivalued(value: Optional[str]) -> Optional[List]:
    """Parse a comma-separated string into a list."""
    if not value:
        return None
    return value.split(',') if isinstance(value, str) and ',' in value else [value]

mode_option = click.option(
    "--mode",
    "-m",
    help="The mode used for analysis."
)
function_option = click.option(
    "--function",
    "-f",
    help="Function within a specified mode."
)
stimulus_option = click.option(
    "--stimuli",
    "-s",
    callback=parse_multivalued,
    help="A comma separated list of stimuli for use of the functions."
)
workdir_option = click.option(
    "--workdir",
    "-w",
    default=Path.cwd,
    show_default=False,
    help="The working directory for the project (defaults to the current directory).",
)