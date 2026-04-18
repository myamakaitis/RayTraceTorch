"""
gui/project.py
--------------
Backward-compatibility shim. The real project I/O lives in
``RayTraceTorch.project`` so it can be used without Dear PyGui installed.
"""

from ..project import save_project, load_project, migrate_project  # noqa: F401
