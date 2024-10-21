from transformers import Trainer, AdamW

class CRFTrainer(Trainer):
    def create_optimizer(self):
        # Use the wrapped model for optimization in case of distributed/sagemaker setups
        opt_model = self.model_wrapped if self.model_wrapped is not None else self.model
        
        if self.optimizer is None:
            # Get parameters that should receive weight decay
            decay_parameters = self.get_decay_parameter_names(opt_model)
            
            # Separate the BERT and CRF layers
            bert_params = list(opt_model.bert.named_parameters())
            classifier_params = list(opt_model.classifier.named_parameters())
            crf_params = list(opt_model.crf.named_parameters())

            # Group the parameters for BERT with decay and no decay
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in bert_params if n in decay_parameters and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,  # Custom learning rate for BERT layers
                },
                {
                    "params": [p for n, p in bert_params if n not in decay_parameters and p.requires_grad],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,  # Custom learning rate for BERT layers
                },
                # Group the parameters for CRF and classifier with decay and no decay
                {
                    "params": [p for n, p in classifier_params + crf_params if n in decay_parameters and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * 3,  # Custom learning rate for CRF and classifier
                },
                {
                    "params": [p for n, p in classifier_params + crf_params if n not in decay_parameters and p.requires_grad],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate * 3,  # Custom learning rate for CRF and classifier
                }
            ]

            # Get optimizer class and kwargs from args (this is generalizable)
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            
            # Handle potential overwrites for specific optimizers
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")
            
            # Initialize AdamW optimizer with the custom parameter groups
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
