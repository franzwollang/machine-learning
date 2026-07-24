# Dataset Fixtures

Port synthetic generators here from `code/legacy/proteus_v1/tests/` and extend them to cover the current SI test plan:

- circles and Swiss rolls;
- nested spheres;
- linked tori;
- variable-density sheets;
- mixed intrinsic dimensions;
- dimensionality junctions;
- the manifold zoo (`manifold_zoo.py`): a connected circle + segment + plane +
  box scene of intrinsic dims {1, 1, 2, 3} meeting at 1<->1 / 1<->2 / 2<->3
  junctions (classic GNG benchmark, OPEN_ISSUES #26). Ships as a diagnostic
  fixture; junction-detection (S8.4) and heterogeneous simplex-dimension (S4.2)
  scenario assertions are deferred until those modules land.
