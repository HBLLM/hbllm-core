import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


def test_onnx():
    print("Downloading model...")
    model_path = hf_hub_download(repo_id="Xenova/paraphrase-MiniLM-L3-v2", filename="onnx/model_quantized.onnx")

    print("Downloading tokenizer...")
    tokenizer_path = hf_hub_download(repo_id="Xenova/paraphrase-MiniLM-L3-v2", filename="tokenizer.json")

    print("Loading...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=128)
    tokenizer.enable_padding(length=128)

    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    input_names = [i.name for i in session.get_inputs()]
    print("Expected Model Inputs:", input_names)

    text = "Calculate the integral of this function."
    enc = tokenizer.encode(text)

    ort_inputs = {
        "input_ids": np.array([enc.ids], dtype=np.int64),
        "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
    }

    if "token_type_ids" in input_names:
        ort_inputs["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)

    outputs = session.run(None, ort_inputs)

    last_hidden_state = outputs[0]
    attention_mask = ort_inputs["attention_mask"].astype(np.float32)
    expanded_mask = np.expand_dims(attention_mask, axis=-1)

    sum_embeddings = np.sum(last_hidden_state * expanded_mask, axis=1)
    sum_mask = np.clip(np.sum(expanded_mask, axis=1), a_min=1e-9, a_max=None)

    sentence_emb = sum_embeddings / sum_mask
    sentence_emb = sentence_emb / np.linalg.norm(sentence_emb, axis=1, keepdims=True)

    print("Success! Final dimension:", sentence_emb.shape)

if __name__ == "__main__":
    test_onnx()
