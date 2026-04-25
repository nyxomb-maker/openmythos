import json

class ModelConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 32000)
        self.hidden_size = kwargs.get("hidden_size", 4096)
        self.intermediate_size = kwargs.get("intermediate_size", 14336)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 32)
        self.num_attention_heads = kwargs.get("num_attention_heads", 32)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 8)  # GQA
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 65536)  # 64k context original
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        
        # MoE specs
        self.num_experts = kwargs.get("num_experts", 64)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 8)
        
        # Memory specs (Per-Layer Compression)
        self.compression_dim = kwargs.get("compression_dim", 256)

    @classmethod
    def from_pretrained(cls, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        return cls(**data)
