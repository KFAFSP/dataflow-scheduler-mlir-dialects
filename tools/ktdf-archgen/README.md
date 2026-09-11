# `ktdf-archgen`

The **ktdf-archgen** tool is a stand-alone executable that **checks** and **bakes** architecture specifications in the `ktdf_arch` dialect.

```
OVERVIEW: ktdf-arch architecture generator

This program runs a checking pipeline on the given input MLIR file (text or bytecode) and, if -o is given, produces a baked architecture specification from it.

USAGE: ktdf-archgen [options] <input file>

OPTIONS:

Generic Options:

  --help          - Display available options (--help-hidden for more)
  --help-list     - Display list of available options (--help-list-hidden for more)
  --version       - Display the version of this program

Tool Options:

  --emit-bytecode - Emit bytecode when generating output
  -o <filename>   - Output filename, omit for checking only
```

The tool always run the checking pipeline, which may be followed by the baking pipeline if `-o <filename>` is specified.

> `ktdf-archgen` also supports the `--debug`, `--debug-only` and `--verify-diagnostics` CLI options with the usual semantics. Run `ktdf-archgen --help-hidden` to get a full list of debugging options.

## Checking

Checking performs the following actions:

- Parses the input file.
- Verifies the input file.
- Runs additional semantic checks (TODO).

If any of these steps fails, the input is rejected and the tool returns with a non-zero status code.

## Baking

Baking always runs after checking and performs the following actions:

- Instantiates all neighborhoods.

If any of these steps fails, the output is discarded and the tool returns with a non-zero status code.
