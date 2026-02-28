import os
import json
import pytest

from hbllm.data.synthesizer import DataSynthesizer

def test_data_synthesizer_generation(tmp_path):
    # Dummy model and tokenizer for test
    class DummyModel:
        pass
    class DummyTokenizer:
        pass
        
    synthesizer = DataSynthesizer(model=DummyModel(), tokenizer=DummyTokenizer())
    
    topic = "astrophysics"
    num_samples = 3
    
    # Use pytest tmp_path for output
    output_dir = str(tmp_path / "synthetic")
    filepath = synthesizer.generate_dataset(topic=topic, num_samples=num_samples, output_dir=output_dir)
    
    # Verify file was created
    assert os.path.exists(filepath)
    
    # Verify contents
    dataset = []
    with open(filepath, "r") as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
            
    assert len(dataset) == num_samples
    
    # Check that topic was injected
    assert topic in dataset[0]["instruction"]
    assert dataset[0]["topic"] == topic
    assert "[Synthetic Data Fragment" in dataset[0]["response"]
