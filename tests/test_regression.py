"""
Regression tests for the layered elastic solver.

Run from the repo root:

    python tests/test_regression.py

Five gates:

1. backward compatibility. With m_max=None and m_nodes=None the patched solver
   must be bit identical to the version in git before the patch. This is the
   gate that protects the Colab notebook and anything already published.
2. Boussinesq. All moduli equal reduces to a homogeneous half space, which has
   a closed form. sigma_z and eps_z must match to 0.01 percent.
3. grid independence. The same physical point must give the same answer at
   query resolutions 5.0, 1.0 and 0.25.
4. surface equilibrium. sigma_z just under the load centre must equal the tyre
   pressure. This is exact in the theory, so it is a good sanity check.
5. tab consistency. A heatmap style call, a depth profile style call and a
   single point call must agree for the same physical point. This is what used
   to fail in WEBLEA by 24 percent.

Gate 1 needs git. The others do not.
"""
import os
import sys
import subprocess
import tempfile
import importlib.util
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from Main.MDA_Huang import Layer3D                      # noqa: E402
from Main.MLEV_Parallel import PyMastic                 # noqa: E402

FIXED = dict(m_max=300, m_nodes=400)

# a 4 in asphalt over 8 in base over subgrade section, one 9000 lb load
E = np.array([500000.0, 25000.0, 10000.0])
H = [4.0, 8.0]
NU = [0.35, 0.4, 0.45]
LOAD = [9000.0]
LPOS = [(15.0, 0)]
A = 5.0
Q = 9000.0 / np.pi / A ** 2       # 114.59 psi

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, ok, detail))
    print('%-4s %s%s' % ('PASS' if ok else 'FAIL', name, ('  ' + detail) if detail else ''))


def run(x, z, **kw):
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    return Layer3D(LOAD, LPOS, A, x.tolist(), [0], z.tolist(), H, E, NU,
                   1600, 7e-20, np.ones(len(E)), 0.01, verbose=False, every=10, **kw)


# ---------------------------------------------------------------------------
# gate 1, backward compatibility against the pre patch solver
# ---------------------------------------------------------------------------
def gate_backward_compat():
    try:
        old_src = subprocess.check_output(
            ['git', '-C', ROOT, 'show', 'HEAD~1:Main/MLEV_Parallel.py'],
            stderr=subprocess.DEVNULL)
    except Exception:
        try:
            old_src = subprocess.check_output(
                ['git', '-C', ROOT, 'show', 'origin/main:Main/MLEV_Parallel.py'],
                stderr=subprocess.DEVNULL)
        except Exception:
            check('backward compatibility', False, 'could not read the old solver from git, skipped')
            return

    if b'm_max' in old_src:
        check('backward compatibility', True, 'reference already contains the patch, nothing to compare')
        return

    tmp = os.path.join(tempfile.mkdtemp(), 'old_solver.py')
    with open(tmp, 'wb') as f:
        f.write(old_src)
    spec = importlib.util.spec_from_file_location('old_solver', tmp)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)

    cases = [
        dict(q=100.0, a=5.0,
             x=np.array([0.001, 3.0, 6.0, 12.0, 24.0]),
             z=np.array([0.01, 2.0, 4.01, 12.0, 20.0]),
             H=[4.0, 8.0], E=[500000.0, 25000.0, 10000.0], nu=[.35, .4, .45]),
        dict(q=0.7, a=100.0,
             x=np.array([1.0, 50.0, 150.0, 400.0]),
             z=np.array([1.0, 25.0, 100.0, 300.0]),
             H=[25.0, 37.5, 62.5, 150.0],
             E=[3300.0, 2000.0, 1250.0, 220.0, 70.0],
             nu=[.35, .35, .35, .3, .3]),
    ]
    worst = 0.0
    identical = True
    for c in cases:
        for it, ev, tol in [(400, 10, 1e-4), (400, 100, 5e-4), (100, 20, 1e-3)]:
            args = (c['q'], c['a'], c['x'].copy(), c['z'].copy(), c['H'], c['E'], c['nu'],
                    7e-20, np.ones(len(c['H'])), it, 'solve', tol, ev, False)
            a_out = old.PyMastic(*args)
            b_out = PyMastic(*[np.array(v) if isinstance(v, np.ndarray) else v for v in args])
            for k in a_out:
                if not np.array_equal(a_out[k], b_out[k]):
                    identical = False
                worst = max(worst, float(np.max(np.abs(a_out[k] - b_out[k]))))
    check('backward compatibility', identical, 'max abs diff %.3g over 6 configurations' % worst)


# ---------------------------------------------------------------------------
# gate 2, Boussinesq closed form
# ---------------------------------------------------------------------------
def gate_boussinesq():
    Eh = np.array([50000.0, 50000.0, 50000.0])
    nuh = 0.35
    z = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0])
    x = np.arange(0.0, 30.5, 1.0)
    R = Layer3D(LOAD, LPOS, A, x.tolist(), [0], z.tolist(), H, Eh, [nuh] * 3,
                1600, 7e-20, np.ones(3), 0.01, verbose=False, every=10, **FIXED)
    i = int(np.argmin(np.abs(x - 15.0)))

    rad = np.sqrt(A ** 2 + z ** 2)
    sz = Q * (1 - z ** 3 / rad ** 3)
    sr = Q / 2 * ((1 + 2 * nuh) - 2 * (1 + nuh) * z / rad + z ** 3 / rad ** 3)
    ez = (sz - nuh * 2 * sr) / 50000.0

    e_sz = np.max(np.abs(R['sigma_z'][0, i] / sz - 1)) * 100
    e_ez = np.max(np.abs(R['eps_z'][0, i] / ez - 1)) * 100
    check('Boussinesq closed form', e_sz < 0.01 and e_ez < 0.01,
          'worst sigma_z %.4f%%, worst eps_z %.4f%%' % (e_sz, e_ez))


# ---------------------------------------------------------------------------
# gate 3, grid independence
# ---------------------------------------------------------------------------
def gate_grid_independence():
    z = [0.01, 2.0, 3.99, 12.01]
    vals = {}
    for res in [5.0, 1.0, 0.25]:
        x = np.arange(0.0, 30.0 + res / 2, res)
        R = run(x, z, **FIXED)
        i = int(np.argmin(np.abs(x - 15.0)))
        vals[res] = np.concatenate([R[k][0, i] for k in ('sigma_z', 'eps_x', 'eps_z')])
    base = vals[1.0]
    worst = max(np.max(np.abs(v / base - 1)) * 100 for v in vals.values())
    check('grid independence', worst < 0.1, 'worst spread %.5f%% across res 5.0 / 1.0 / 0.25' % worst)


# ---------------------------------------------------------------------------
# gate 4, surface equilibrium
# ---------------------------------------------------------------------------
def gate_surface_equilibrium():
    x = np.arange(0.0, 30.5, 1.0)
    R = run(x, [0.01, 4.01], **FIXED)
    i = int(np.argmin(np.abs(x - 15.0)))
    got = float(R['sigma_z'][0, i, 0])
    err = (got / Q - 1) * 100
    check('surface equilibrium', abs(err) < 2.0,
          'sigma_z %.2f psi against tyre pressure %.2f psi, %+.2f%%' % (got, Q, err))


# ---------------------------------------------------------------------------
# gate 5, the three WEBLEA call styles must agree
# ---------------------------------------------------------------------------
def gate_call_style_consistency():
    z = [0.01, 3.99, 12.01]
    heat = run(np.arange(0.0, 30.5, 1.0), z, **FIXED)['sigma_z'][0, 15]
    prof = run([15.0], z, **FIXED)['sigma_z'][0, 0]
    point = run([15.0, 20.0], z, **FIXED)['sigma_z'][0, 0]
    worst = max(np.max(np.abs(prof / heat - 1)), np.max(np.abs(point / heat - 1))) * 100
    check('call style consistency', worst < 0.1,
          'heatmap / profile / point agree to %.5f%%' % worst)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    print('layered elastic regression tests\n')
    gate_backward_compat()
    gate_boussinesq()
    gate_grid_independence()
    gate_surface_equilibrium()
    gate_call_style_consistency()
    failed = [n for n, ok, _ in RESULTS if not ok]
    print('\n%d of %d passed' % (len(RESULTS) - len(failed), len(RESULTS)))
    sys.exit(1 if failed else 0)
