"""Legacy module path for historical commands. New code uses palimpsest.

Only the namespace is forwarded; no scientific implementation is duplicated.
"""

from palimpsest import simulation as _simulation

__path__ = _simulation.__path__
