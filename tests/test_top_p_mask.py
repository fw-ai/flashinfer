import pytest
import torch
from flashinfer.sampling import top_p_mask, top_p_renorm_probs


@pytest.mark.parametrize("rows,vocab", [(1, 2048), (4, 4097), (6, 163840), (16, 65536)])
@pytest.mark.parametrize(
    "kind", ["random", "peaked", "uniform", "sparse", "zero", "tiny", "rounding"]
)
def test_mask_matches_flashinfer(rows, vocab, kind):
    torch.manual_seed(123)
    p = torch.randn(rows, vocab, device="cuda").softmax(-1)
    if kind == "peaked":
        p = (torch.randn_like(p) * 10).softmax(-1)
    elif kind == "uniform":
        p.fill_(1 / vocab)
    elif kind == "sparse":
        p.zero_()
        p[:, :4] = torch.tensor([0.5, 0.25, 0.125, 0.125], device="cuda")
    elif kind == "zero":
        p.zero_()
    elif kind == "tiny":
        p.zero_()
        p[:, 0] = 1
        # Exercise FP32 normal/subnormal boundary and FlashInfer's FTZ multiply.
        p[:, 1:5] = torch.tensor(
            [1e-45, 1e-38, torch.finfo(torch.float32).tiny, 1e-30], device="cuda"
        )
    elif kind == "rounding":
        p.zero_()
        p[:, :2] = 0.50000006
        p[:, 2] = torch.finfo(torch.float32).tiny
        p[:, 3] = torch.finfo(torch.float32).tiny * 2
    for cutoff in [0.0, 0.5, 0.9, 0.999999, 1.0]:
        top_p = torch.full((rows,), cutoff, device="cuda")
        actual = top_p_mask(p, top_p, is_deterministic=True)
        assert actual is not None
        expected = top_p_renorm_probs(p, top_p, is_deterministic=True) == 0
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_mixed_thresholds_and_graph_replay():
    p = torch.randn(6, 163840, device="cuda").softmax(-1)
    top_p = torch.tensor([0.0, 0.1, 0.5, 0.9, 0.99, 1.0], device="cuda")
    top_p_mask(p, top_p, is_deterministic=True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = top_p_mask(p, top_p, is_deterministic=True)
    for _ in range(3):
        p.copy_(torch.randn_like(p).softmax(-1))
        top_p.copy_(top_p.roll(1))
        graph.replay()
        torch.testing.assert_close(
            actual,
            top_p_renorm_probs(p, top_p, is_deterministic=True) == 0,
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("rows,vocab", [(3, 127), (65, 2048), (0, 2048), (3, 0)])
def test_fallback_and_empty(rows, vocab):
    p = torch.randn(rows, vocab, device="cuda").softmax(-1)
    actual = top_p_mask(p, 0.9, True)
    assert actual.dtype == torch.bool and actual.shape == p.shape
    if p.numel():
        torch.testing.assert_close(actual, top_p_renorm_probs(p, 0.9, True) == 0)


def test_layout_and_dtype():
    p = torch.randn(3, 4096, device="cuda").softmax(-1).half().float()
    p = (p / p.sum(-1, keepdim=True)).half()
    p = p.t().contiguous().t()
    threshold = torch.tensor([0.1, 0.5, 0.9], device="cuda", dtype=torch.float64)
    torch.testing.assert_close(
        top_p_mask(p, threshold, True),
        top_p_renorm_probs(p.float().contiguous(), threshold.float(), True) == 0,
    )


def test_invalid_inputs():
    with pytest.raises(ValueError):
        top_p_mask(torch.ones(1, 2048), 0.9)
    p = torch.randn(3, 2048, device="cuda").softmax(-1)
    with pytest.raises(ValueError):
        top_p_mask(p, torch.ones(2, device="cuda"))
    with pytest.raises(ValueError):
        top_p_mask(p, torch.ones(3))


def test_default_radix_mode():
    p = torch.zeros(6, 163840, device="cuda")
    p[:, :4] = torch.tensor([0.5, 0.25, 0.125, 0.125], device="cuda")
    torch.testing.assert_close(top_p_mask(p, 0.9), top_p_renorm_probs(p, 0.9) == 0)


def test_trace_output_dtype():
    p = torch.empty(3, 4097)
    definition = top_p_mask.fi_trace(probs=p, top_p=0.9)
    assert definition["outputs"]["mask"]["dtype"] == "bool"
