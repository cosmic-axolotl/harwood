"""The documentation site: built from docsite/pages and from the tasks
themselves, read in the window and in a browser -- the same files in both.
See docsite/builder.py."""

from .builder import BuildError, build, ensure, location, url_for

__all__ = ["BuildError", "build", "ensure", "location", "url_for"]
