# Route Capture Patch

`Dockerfile.route-capture` builds the formal GLM-5.2 W8A8 capture image from
the official `quay.io/ascend/vllm-ascend:v0.22.1rc1` base. The build invokes
`apply_w8a8_route_capture.py`, which has three hard guards:

- installed packages must be vLLM `0.22.1` and vLLM-Ascend `0.22.1rc1`;
- the unmodified `w8a8_dynamic.py` must match SHA-256
  `1dd59f6f8114e19824d559b99cc4a22fed04e54ff0ecd9e853aa3b6a574699e2`;
- the hook must appear exactly once before zero-expert handling and force load
  balancing.

The hook calls the `layer.router.capture_fn` already bound by vLLM when
`--enable-return-routed-experts` is enabled. It therefore sends logical W8A8
`topk_ids` to vLLM's existing capture buffer before any subsequent mapping can
change their meaning. The image receives the Docker label
`glm52.capture_patch_id=glm52-w8a8-logical-topk-v1`.

Build only with:

```bash
bash scripts/07_build_capture_image.sh --confirm-pull-base
```

The image build checks the package versions, label, and inserted marker. It is
still not enough to establish scientific validity: after the server is started,
`scripts/12_smoke_request.py --require-routes` must pass before running a
benchmark suite.
