from transformers import AutoTokenizer, AutoModel
from lavis.models.base_model import BaseModel

class BaseTextModel(BaseModel):
    def __init__(self, pretrained_model_name_or_path, use_local=False):
        super().__init__()
        self.load_tokenizer(pretrained_model_name_or_path, use_local)
        self.load_model(pretrained_model_name_or_path, use_local)
    
    def load_tokenizer(self, pretrained_model_name_or_path, use_local):
        if use_local:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    def load_model(self, pretrained_model_name_or_path, use_local):
        if use_local:
            self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        else:
            self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
    
    @classmethod
    def from_config(cls, cfg):
        pretrained_model_name_or_path = cfg.get("pretrained_model_name_or_path", "BAAI/bge-large-en-v1.5")
        use_local = cfg.get("use_local", False)
        return cls(pretrained_model_name_or_path, use_local)