"""BirdGate — an always-on birdsong gate for the AudioMoth.

Binary presence detection (bird / no bird) via a template-SVM over a CAR-IHC
cochlea front-end, sized to run continuously on a Cortex-M4. See docs/lab-notebook.md.
"""
__all__ = ["config", "data", "windows", "carfac", "frontends", "gate", "metrics"]
