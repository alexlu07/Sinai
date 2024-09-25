from dataclasses import dataclass
from typing import Optional, Tuple, Union
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import ModelOutput
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

@dataclass
class CRFClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    emissions: Optional[torch.FloatTensor] = None

class BertForTokenClassificationWithCRF(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)  # Define CRF layer

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CRFClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        crf_mask = attention_mask.bool()

        loss = None
        if labels is not None:
            # Calculate the negative log-likelihood
            loss = -self.crf(emissions, labels, mask=crf_mask)

        # Perform Viterbi decoding to get the most likely labels
        decoded_tags = self.crf.decode(emissions, mask=crf_mask)
        decoded_tags = torch.tensor(decoded_tags, dtype=torch.long, device=emissions.device)
        logits = F.one_hot(decoded_tags, num_classes=self.num_labels).to(emissions.dtype) # scuffed solution

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CRFClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            emissions=emissions
        )