# Grid independent Hankel integration

July 2026. Notes on why `m_max` and `m_nodes` were added to `MLEV_Parallel.PyMastic`.

## Short version

Results used to depend on how many query points you asked for. Add more points to the grid and the near surface stresses changed, even though the structure and the load did not. Everything at and below the bottom of the asphalt layer was fine. The top inch or two was not.

Two new optional arguments fix it. Both default to `None`, and with the defaults the solver behaves exactly as before, bit for bit. Set `m_max=300, m_nodes=400` to get the corrected path.

## Problem 1, the integral was truncated by an index

`PyMastic` evaluates a Hankel integral by Gauss quadrature over a variable `m`. The quadrature nodes come from a pooled list of Bessel zeros:

```python
firstKindZeroOrder = firstKindZeroOrder / ro[:, None]   # ro = x / sum(H), one column per query radius
BesselZeros = np.sort(np.hstack((0, firstKindZeroOrder.flatten(), firstKindFirstOrder.flatten())))
mValues = np.hstack((AUX1, AUX2[1:], BesselZeros[3:iteration]))
```

The 100 zeros of J0 get divided by every query radius, then pooled and sorted. Cutting that pooled list at index `iteration` cuts the integral at a different value of `m` depending on how many radii went in. More query points make the pool denser, so a fixed index reaches a lower `m`, so the integral is worse.

Measured on a 4 in over 8 in section with `iteration=400`, which is what WEBLEA's `iterations: 1600` becomes after `it = int(it/4)`:

| resolution | radii passed | m reached |
|---|---|---|
| 5.0 | 4 | 501 |
| 1.0 | 16 | 81 |
| 0.5 | 31 | 43 |
| 0.25 | 61 | 22.5 |
| 0.1 | 251 | 6.3 |

The integral needs `m` up to about 300 before the surface stress settles.

## Problem 2, convergence was measured on deflection only

The `every` step check computed only `displacementZ` and compared it to the previous check. Deflection is the fastest converging quantity in the set. Its integrand is dominated by small `m`, while the near surface stresses need large `m`. So the loop returned as soon as the deflection settled and the stress sums were silently cut short:

| resolution | stopped at | m at that point | m available on the grid |
|---|---|---|---|
| 5.0 | j = 90 of 1620 | 18.97 | 751.8 |
| 1.0 | j = 130 | 9.44 | 121.7 |
| 0.25 | j = 340 | 7.70 | 33.8 |

Every case cut off at `m` between 8 and 19. The two problems compound. Problem 2 stops early, problem 1 caps how far it could have gone anyway.

There was no way to fix this from the settings. Raising `every` helps on coarse grids and makes things worse on fine ones, because on a fine grid the node list no longer reaches far enough:

| | every=10 | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|---|
| surface sigma_z error, res 1.0 | +41.3% | +21.5% | +8.7% | -9.8% | -7.7% | -5.6% |
| surface sigma_z error, res 0.25 | +37.6% | +39.7% | +24.1% | -17.6% | +41.4% | +41.4% |

Raising `iteration` did nothing at all, for the reason in the next section.

## Problem 3, the bare except hid all of this

`MDA_Huang.Layer3D` had this:

```python
try:
    DRS[i] = PyMastic(..., every=every, ...)
except:
    DRS[i] = PyMastic(..., every=20, ...)
```

When the run did not converge, `PyMastic` raised `TypeError`, this caught it, and the retry used `every=20`. That converges almost immediately at very low `m`, so the fallback was the least accurate setting available. Nothing was printed.

This made `iteration` inert through `Layer3D`. Raising it from 1600 to 14256 at resolution 0.25 took the runtime from 1.1 s to 7.3 s and did not change a single digit of the answer, because both runs ended up in the fallback.

It also swallowed `KeyboardInterrupt`, `MemoryError` and real `LinAlgError`s. It is now `except TypeError` with a warning.

## What the errors actually were

4 in asphalt over 8 in base over subgrade, one 9000 lb load, a = 5 in, `iterations=1600`, `tolerance=0.01`, `every=10`. Error against a grid independent reference, under the load:

| resolution | surface sigma_z | z=2 in sigma_z | bottom of AC sigma_z | z=12 in sigma_z | bottom of AC eps_x |
|---|---|---|---|---|---|
| 5.0 | -16.4% | -5.5% | -0.11% | 0.00% | +0.2% |
| 2.5 | -14.8% | -3.3% | +0.04% | 0.00% | -0.1% |
| 1.0 | +41.3% | +26.0% | +2.2% | 0.00% | -3.9% |
| 0.5 | +40.9% | +25.8% | +2.1% | 0.00% | -3.8% |
| 0.25 | +33.5% | +20.6% | +1.3% | -0.01% | -2.4% |

Three things a user would notice:

1. The top row of a heatmap moved when you changed the resolution.
2. Surface sigma_z came out as 160.5 psi under a 114.6 psi tyre. Surface vertical stress under a uniform circular load has to equal the tyre pressure, so that is visibly wrong.
3. In WEBLEA the Heatmap tab and the Depth profile tab disagreed for the same physical point, 160.5 psi against 129.5 psi. The heatmap passes 16 radii and the depth profile passes 1.

The Colab notebook was in much better shape. At its Step 2 settings everything at and below the bottom of the asphalt was correct to five significant figures. Only the z = 0 row was wrong, by -25% under the load and +35% between the loads.

## The fix

`m_max` truncates on the value of `m` instead of on a list index. `m_nodes` replaces the pooled zeros with a uniform grid of nodes over `[0, m_max]`. When `m_max` is set the whole fixed grid is integrated and the adaptive stop is skipped, since a fixed grid does not need one.

`Layer3D` passes both straight through. Both default to `None` everywhere.

## Verification

Backward compatibility. With `m_max=None, m_nodes=None` the patched and the old solver give bit identical output. `np.array_equal` is true and the maximum absolute difference is exactly 0.0, across two structures (imperial 4/8 in, SI 25/37.5/62.5/150 mm) and three settings of `iteration`, `every` and `tol`.

Accuracy. Setting all moduli equal reduces the problem to a Boussinesq half space, which has a closed form. With `m_max=300, m_nodes=400`, sigma_z and eps_z match to 0.001% at depths from 0.5 in to 30 in.

Grid independence. With the new path the answer is identical to five decimal places at resolutions 5.0, 1.0 and 0.25, and identical between `m_nodes` of 400 and 1200. It is also now independent of `every`.

Speed. The old pooled zeros oversample badly and make the cost quadratic in the number of radii.

| resolution | n_x | before | after | speed up |
|---|---|---|---|---|
| 1.0 | 31 | 0.26 s | 0.25 s | 1.05x |
| 0.5 | 61 | 0.60 s | 0.32 s | 1.86x |
| 0.25 | 121 | 1.68 s | 0.49 s | 3.44x |
| 0.1 | 301 | 12.38 s | 1.62 s | 7.63x |

## Known limit

`m_max` cannot go much above 300. At 400 the layer matrix recursion overflows to NaN and at 3000 it raises a singular matrix. The ceiling is the same for sum(H) of 12 in, 48 in and 300 mm, because the depths are normalised by sum(H).

At `m_max=300` the surface sigma_z still carries about -0.8% residual error, and it does not approach that monotonically (100 gives +12.3%, 200 gives -5.6%, 300 gives -0.8%). So this takes the surface from plus or minus 40% down to about 1%, not to exact. Going further would mean clipping the exponentials in the layer recursion, which is a much bigger change and is not worth it.

## Files with their own copy of the solver

`webapp/` on the WEBLEA branch keeps its own copies of `MLEV_Parallel.py` and `MDA_Huang.py` so that Cloud Run can use a flat layout. They are identical to the ones in `Main/` apart from the import line. Any future solver change has to go into both.
