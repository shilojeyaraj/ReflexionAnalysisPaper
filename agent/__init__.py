"""
Agent components for the Reflexion memory study.

Actor: generates task responses informed by past reflections from memory.
Reflector: generates concrete lessons learned from each attempt.
run_trial_loop: orchestrates the full trial loop for a single task.
"""

from agent.actor import Actor
from agent.reflector import Reflector
from agent.loop import run_trial_loop

__all__ = ["Actor", "Reflector", "run_trial_loop"]
