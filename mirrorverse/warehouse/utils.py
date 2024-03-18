"""
Utility functions for the mirrorverse warehouse.
"""

from sqlalchemy import create_engine


def get_engine():
    return create_engine("sqlite://", echo=True)
