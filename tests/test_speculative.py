import torch
import torch.nn as nn

from hbllm.model.speculative import speculate_step


class MockModel(nn.Module):
    def __init__(self, vocab_size=10, always_predict=5):
        super().__init__()
        self.vocab_size = vocab_size
        self.always_predict = always_predict

    def forward(self, x, **kwargs):
        batch, seq_len = x.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size)
        # Put 100% logic on the always_predict token to make probabilities deterministic
        logits[:, :, self.always_predict] = 100.0
        return {"logits": logits}

def test_speculative_equivalence():
    """
    Test that speculative decoding mathematically preserves target model outputs
    and resamples correctly on rejection.
    """
    # 1. Same logic - Draft and Main always agree
    draft_model = MockModel(vocab_size=10, always_predict=3)
    main_model = MockModel(vocab_size=10, always_predict=3)

    input_ids = torch.tensor([[1, 2]])

    # 2 tokens generated + 1 free token from perfect approval
    out, _, _ = speculate_step(main_model, draft_model, input_ids, input_ids, K=2)
    assert out.shape == (1, 3)
    assert out[0, 0].item() == 3
    assert out[0, 1].item() == 3
    assert out[0, 2].item() == 3

    # 2. Complete Rejection - Draft predicts 4, Main predicts 7
    draft_model = MockModel(vocab_size=10, always_predict=4)
    main_model = MockModel(vocab_size=10, always_predict=7)

    out, _, _ = speculate_step(main_model, draft_model, input_ids, input_ids, K=2)
    # The first token is drafted as 4, but target forces it to 7.
    # Because target probability for 4 is 0.0, the draft is rejected on token 0 constraint,
    # and we get exactly 1 token (the resampled correct token).
    assert out.shape == (1, 1)
    assert out[0, 0].item() == 7

