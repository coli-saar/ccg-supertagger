import sys

from torch import nn
from torch.nn import Linear
from transformers import RobertaModel, XLMRobertaModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class TaggingModel(nn.Module):
    def __init__(self, output_dim, roberta_hidden_dim=768, roberta_id="xlm-roberta-base"):
        super().__init__()

        # https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
        self.roberta = XLMRobertaModel.from_pretrained(roberta_id)
        self.linear = Linear(roberta_hidden_dim, output_dim)

        # freeze Roberta parameters
        for name, param in self.named_parameters():
            if name.startswith("roberta"):
                param.requires_grad = False

    def forward(self, x, attention_mask):
        # x: LongTensor (bs, seqlen)
        out: BaseModelOutputWithPoolingAndCrossAttentions = self.roberta(x, attention_mask=attention_mask)
        hidden_states = out.last_hidden_state  # (bs, seqlen, hidden_size)
        logits = self.linear(hidden_states) # (bs, seqlen, output_dim)
        return logits

