"""Synthetic Verifiers rubric artifact.

Adapter tests must never import or execute this file. It exists only so artifact
preservation logic has a nearby Python rubric file to reference.
"""

async def correct_answer(completion, answer):
    raise RuntimeError("fixture rubric.py was executed")
