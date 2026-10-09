import math

import torch

from ics.optim import AdamATan2, SparseSignSGD
from ics.trm.layers import CastedSparseEmbedding


def test_adam_atan2_follows_its_update_rule():
    torch.manual_seed(0)
    p0, grads = torch.randn(5, 3), [torch.randn(5, 3) for _ in range(3)]
    p = torch.nn.Parameter(p0.clone())
    opt = AdamATan2([p], lr=0.01, betas=(0.9, 0.95), weight_decay=0.1)
    w, m, v = p0.double(), torch.zeros(5, 3, dtype=torch.float64), torch.zeros(5, 3, dtype=torch.float64)
    for t, g in enumerate(grads, 1):
        p.grad = g.clone()
        opt.step()
        m, v = 0.9 * m + 0.1 * g.double(), 0.95 * v + 0.05 * g.double() ** 2
        w = w * (1 - 0.01 * 0.1) - 0.01 / (1 - 0.9 ** t) * torch.atan2(m, v.sqrt() / math.sqrt(1 - 0.95 ** t))
    torch.testing.assert_close(p.detach().double(), w, rtol=1e-5, atol=1e-6)
    assert opt.state[p]["step"] == 3


def test_sparse_sign_sgd_updates_the_rows_looked_up_by_every_micro_batch():
    grads = torch.tensor([[1.0, -2.0, 0.5], [3.0, 1.0, -1.0], [-2.0, 1.0, 0.25], [0.5, 0.5, 0.5]])
    ids = torch.tensor([2, 0, 2, 3], dtype=torch.int32)

    def stepped(micro: int) -> torch.Tensor:
        emb = CastedSparseEmbedding(4, 3, batch_size=micro, init_std=0.0, cast_to=torch.float32).train()
        with torch.no_grad():
            emb.weights.copy_(torch.arange(12.0).view(4, 3))
        opt = SparseSignSGD(emb, lr=0.1, weight_decay=0.5)
        for s in range(0, 4, micro):                         # each forward overwrites the local rows and ids
            (emb(ids[s:s + micro]) * grads[s:s + micro]).sum().backward()
        opt.step()
        assert opt.pending == [] and emb.local_weights.grad is None
        return emb.weights

    expected = torch.arange(12.0).view(4, 3) * 0.95
    expected[0] -= 0.1 * torch.tensor([1.0, 1.0, -1.0])
    expected[2] -= 0.1 * torch.tensor([-1.0, -1.0, 1.0])             # rows 0 and 2 of the batch, summed
    expected[3] -= 0.1
    expected[1] = torch.arange(3.0, 6.0)                                # never looked up
    torch.testing.assert_close(stepped(2), expected)
    torch.testing.assert_close(stepped(2), stepped(4), rtol=0, atol=0)
