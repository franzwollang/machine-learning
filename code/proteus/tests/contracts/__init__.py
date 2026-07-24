"""Typed I/O contracts matching the Proteus SI specification.

These dataclasses define the public shapes of Proteus objects. The canonical
definitions now live in :mod:`proteus.types` (promoted out of the test tree per
OPEN_ISSUES #38); the modules in this package re-export them, grouped by the SI
section each shape belongs to, so tests and future implementation modules share
one source of truth.
"""
