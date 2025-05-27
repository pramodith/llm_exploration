from transformers import AutoModelForCausalLM, AutoTokenizer
import lightning as pl

class GRPOTrainer(pl.LightningModule):
    def __init__(self, model_name_or_path: str):
        """
        Initialize the GRPOTrainer with a model.

        Args:
            model_name_or_path (str): Path to the model or model identifier 
                from the Hugging Face Model Hub.

        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.policy_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.reference_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.reference_model.eval()