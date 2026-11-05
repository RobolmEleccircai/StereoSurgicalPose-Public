"""Run the frozen collector on Isaac Sim 6, adapting its removed semantic helper.

The collector, geometry, trajectories, camera model, and quality settings stay
unchanged. This adapter has its own recorded SHA-256, separate from plan identity.
"""
import argparse
import hashlib
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--code', type=Path, required=True)
a, capture_args = p.parse_known_args()
sys.path.insert(0, str(a.code))
import isaacsim

NativeApp = isaacsim.SimulationApp
instances = []

class CompatibleApp(NativeApp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_closed = False
        instances.append(self)
        from isaacsim.core.utils import semantics
        if not hasattr(semantics, 'add_update_semantics'):
            def add_update_semantics(prim, semantic_label, type_label='class'):
                semantics.add_labels(prim, [semantic_label], instance_name=type_label, overwrite=True)
            semantics.add_update_semantics = add_update_semantics

    def close(self, *args, **kwargs):
        self.adapter_closed = True
        return super().close(*args, **kwargs)

isaacsim.SimulationApp = CompatibleApp
print('RUNTIME6_SEMANTIC_ADAPTER', hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), flush=True)
sys.argv = [str(a.code / 'capture.py'), *capture_args]
try:
    import capture
    capture.main()
finally:
    for app in instances:
        if not app.adapter_closed:
            app.close(wait_for_replicator=False)
