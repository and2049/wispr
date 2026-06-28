# Test Fixtures

`jingle_bells.m4a` and `jingle_bells.txt` are public-domain test fixtures used
for CI smoke coverage. The normal CI workflow runs them with the mock backend,
so no model downloads or audio decoding are required.

The fixture exists to exercise real file paths, lyrics parsing, benchmark report
generation, and CLI output handling. It is not used as an accuracy benchmark in
the default test suite.
