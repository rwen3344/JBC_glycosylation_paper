#!/usr/bin/env python3
"""install_requirements.py -- install the Python packages needed to run
figure_manuscript_code_standalone.py.

Usage
-----
    python install_requirements.py            # exact versions the figures were generated with
    python install_requirements.py --latest   # newest available versions instead

Packages are installed into the Python interpreter that runs this script
(`sys.executable`), so run it with the same `python` you will use for the
figure code. Using a fresh virtual environment is recommended:

    python -m venv venv
    source venv/bin/activate          # Windows: venv\\Scripts\\activate
    python install_requirements.py

The pinned versions below were tested with Python 3.14.
"""

import importlib
import subprocess
import sys

# pip package name -> (tested version, import name used to verify the install)
PACKAGES = {
    'pandas': ('2.3.3', 'pandas'),
    'numpy': ('2.5.4', 'numpy'),
    'scipy': ('1.18.1', 'scipy'),
    'statsmodels': ('0.15.0', 'statsmodels'),
    'matplotlib': ('3.11.2', 'matplotlib'),
    'seaborn': ('0.13.2', 'seaborn'),
    'scikit-learn': ('1.9.1', 'sklearn'),
    'adjustText': ('1.4.0', 'adjustText'),
    'lifelines': ('0.30.3', 'lifelines'),
    'gseapy': ('1.3.1', 'gseapy'),
    'pillow': ('12.3.0', 'PIL'),
    'openpyxl': ('3.1.5', 'openpyxl'),
}


def main():
    latest = '--latest' in sys.argv[1:]
    specs = [name if latest else f'{name}=={version}' for name, (version, _) in PACKAGES.items()]

    print(f'Installing into: {sys.executable}')
    print('Packages: ' + ', '.join(specs) + '\n')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *specs])

    # verify every package imports
    print('\nChecking installed packages:')
    failed = []
    for name, (_, module) in PACKAGES.items():
        try:
            mod = importlib.import_module(module)
            print(f'  OK      {name} {getattr(mod, "__version__", "")}')
        except Exception as e:
            print(f'  FAILED  {name}: {e}')
            failed.append(name)

    if failed:
        sys.exit(f'\nSome packages failed to import: {", ".join(failed)}')
    print('\nAll packages installed. Run: python figure_manuscript_code_standalone.py')


if __name__ == '__main__':
    main()
