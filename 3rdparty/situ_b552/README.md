# Private SiTU b552 source inputs

This directory vendors the source inputs used to produce Fireworks' isolated
`mxfp8_mxfp4_ll_situ` kernel bundle. They are build inputs and are never used
by the stock FlashInfer d2c runner.

- `pr2917/` contains exactly the 102 MXFP8-activation/MXFP4-weight SM100
  SwiGLU `.cu/.h` pairs from FlashInfer commit
  `64ed071e23bf8d5d2d5af5c91577e5b8e036a1cf`, plus that snapshot's `LICENSE`
  and `NOTICE` files.
- `b552/include/` contains the 18 untouched generated headers from the pinned
  public b552 artifact `batched_gemm-b3c1646-c111d7c`.

`scripts/situ_b552/build.py` verifies aggregate and per-file SHA-256 hashes,
generates the private SiTU children in a temporary build directory, and seals
the compiled cubins and ABI contract into a separate artifact namespace. Do
not edit generated kernel inputs in this directory; update the immutable pins
and signed-off hashes as one reviewed change.
