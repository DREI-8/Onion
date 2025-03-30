import sys
import shutil
from pathlib import Path

site_packages = Path(sys.prefix) / 'Lib' / 'site-packages'
local_site_packages = Path(sys.prefix) / 'local' / 'lib' / 'python3.11' / 'site-packages'
user_site_packages = Path.home() / '.local' / 'lib' / 'python3.11' / 'site-packages'

package_paths = [
    site_packages / 'onion',
    local_site_packages / 'onion',
    user_site_packages / 'onion',
    site_packages / 'onion-0.1.0.dist-info',
    site_packages / 'onion-0.1.0.egg-info',
]

for path in package_paths:
    if path.exists():
        print(f"Suppression de {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

print("Nettoyage terminé")

try:
    import onion
    print("Le package est toujours installé!")
except ImportError:
    print("Le package a été correctement supprimé")

modules = [m for m in sys.modules if m.startswith('onion')]
if modules:
    print(f"Modules onion toujours en mémoire: {modules}")