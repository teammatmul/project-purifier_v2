import logging
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from activations import get_activation
from configuration_electra import ElectraConfig
from file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
    replace_return_docstrings,
)
from modeling_bert import BertEmbeddings, BertEncoder, BertLayerNorm, BertPreTrainedModel
from modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from modeling_utils import SequenceSummary

logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "ElectraConfig"
_TOKENIZER_FOR_DOC = "ElectraTokenizer"

ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/electra-small-generator",
    "google/electra-base-generator",
    "google/electra-large-generator",
    "google/electra-small-discriminator",
    "google/electra-base-discriminator",
    "google/electra-large-discriminator",
    # See all ELECTRA models at https://huggingface.co/models?filter=electra
]


class ElectraEmbeddings(BertEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.embedding_size, eps=config.layer_norm_eps)


class ElectraDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze()

        return logits


class ElectraGeneratorPredictions(nn.Module):
    """Prediction module for the generator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.LayerNorm = BertLayerNorm(config.embedding_size)
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(self, generator_hidden_states):
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = get_activation("gelu")(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class ElectraPreTrainedModel(BertPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = ElectraConfig
    # load_tf_weights = load_tf_weights_in_electra
    base_model_prefix = "electra"


@dataclass
class ElectraForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.ElectraForPreTrainingModel`.
    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss of the ELECTRA objective.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


ELECTRA_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.
    Parameters:
        config (:class:`~transformers.ElectraConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ELECTRA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`transformers.ElectraTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.__call__` for details.
            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the hidden states of all layers are returned. See ``hidden_states`` under returned tensors for more detail.
        return_dict (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the model will return a :class:`~transformers.file_utils.ModelOutput` instead of a
            plain tuple.
"""


@add_start_docstrings(
    "The bare Electra Model transformer outputting raw hidden-states without any specific head on top. Identical to "
    "the BERT model except that it uses an additional linear layer between the embedding layer and the encoder if the "
    "hidden size and embedding size are different."
    ""
    "Both the generator and discriminator checkpoints may be loaded into this model.",
    ELECTRA_START_DOCSTRING,
)
class ElectraModel(ElectraPreTrainedModel):

    config_class = ElectraConfig

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = ElectraEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)

        self.encoder = BertEncoder(config)
        self.config = config
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(ELECTRA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/electra-small-discriminator",
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)

        hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return hidden_states, extended_attention_mask


class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """ELECTRA Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    ELECTRA_START_DOCSTRING,
)
###### purifier coding
class PuriAttention(nn.Module):
    """This layer is important role for purify toxic expression
    You can select the output of any encoder layer and embedding output to use query, key, value
    and we just use layer normalization when choosing multiple layer output.
    embedding output : 0
    1~12 encoder layer : 1~12

    Params:
        `config` : a BertConfig class instance with the configuration to build a new model.

    Input:
        `query_hidden_states` : average of values passed through the selected layer for query
        `key_hidden_states` : average of values passed through the selected layer for key
        `value_hidden_states` : average of values passed through the selected layer for value
        `attention_mask` : an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `query_att` : choose to use query_attention matrix when update attention_probs. Default: `False`.
        `key_att` : choose to use key_attention matrix when update attention_probs. Default: `False`.
        `multi_head` : choose to apply multi-head attention. Default: `True`.
        `dropout` : choose to apply dropout to attention_probs. Default: `False`.
        `pooler` : choose to apply tanh activation function. Default: `True`.

    Outputs:
        `attention_output` : final computation of cls token
        `cls_info` : attention_scores, attention_probs values passed through puri layer.

    """

    def __init__(self, config):
        super(PuriAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_hidden_states, key_hidden_states, value_hidden_states, attention_mask,
                query_att=False, key_att=False, multi_head=True):
        # to save weight information
        cls_info = {}

        # choose to use query_attention matrix when update attention_probs
        # all of hidden_states sizes are [32, 128, 768] and it belongs to selected encoder layers output
        if query_att:
            mixed_query_layer = self.query(query_hidden_states)
        else:
            mixed_query_layer = query_hidden_states[:, :, :]
        if key_att:
            mixed_key_layer = self.key(key_hidden_states)
        else:
            mixed_key_layer = key_hidden_states[:, :, :]
        mixed_value_layer = value_hidden_states[:, :, :]

        # block attention score cls to itself
        attention_mask[:, :, :, 0] -= 10000

        # apply multi-head attention
        if multi_head:
            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)

            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            # attention_scores size becomes [batch, num_head, 1(cls), seq_length]
            attention_scores = attention_scores[:, :, 0:1, :]
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

            # context_layer size is [batch, num_head, 1(cls), head_size]
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            # context_layer size is [batch, 1(cls), all_head_size]
            attention_output = context_layer.view(*new_context_layer_shape)

            # attention size becomes [batch, 1(cls), num_head, seq_length]
            attention_probs = attention_probs.permute(0, 2, 1, 3).contiguous()
            attention_scores = attention_scores.permute(0, 2, 1, 3).contiguous()

        # apply single-head attention
        else:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(mixed_query_layer, mixed_key_layer.transpose(-1, -2))
            # attention_scores size becomes [batch, 1(cls), seq_length]
            attention_scores = attention_scores[:, 0:1, :]
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            # reduce rank of attention mask
            attention_mask = attention_mask.squeeze(1)
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

            # attention_output size is [batch, 1(cls), all_head_size]
            attention_output = torch.matmul(attention_probs, mixed_value_layer)

        cls_info['scores'] = attention_scores
        cls_info['probs'] = attention_probs

        # # reduce the rank of attention_output
        # attention_output = attention_output[:, 0]

        return attention_output, cls_info


class PurifierModel(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.puri = PuriAttention(config)
        self.classifier = ElectraClassificationHead(config)

        self.init_weights()

    @add_start_docstrings_to_callable(ELECTRA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/electra-small-discriminator",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def select_layers(self, output_layers, selected_layers):
        # selected_layers is indices of index which 0 index means embedding_output
        mean_output_layers = output_layers[selected_layers[0]]
        if len(selected_layers) > 1:
            for idx in selected_layers[1:]:
                mean_output_layers = torch.add(mean_output_layers, output_layers[idx])
            mean_output_layers = self.LayerNorm(mean_output_layers)
        return mean_output_layers

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
        query=[12], key=[1,2,3], value=[1,2,3],
        query_att=True, key_att=True,
        multi_head=False
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states, extended_attention_mask = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        sequence_output = discriminator_hidden_states[1]

        query_hidden_states = self.select_layers(sequence_output, query)
        key_hidden_states   = self.select_layers(sequence_output, key)
        value_hidden_states = self.select_layers(sequence_output, value)

        cls_output, cls_info = self.puri(query_hidden_states, key_hidden_states, value_hidden_states, extended_attention_mask,
                                         query_att, key_att, multi_head,)

        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if not return_dict:
                output = (logits,) + discriminator_hidden_states[1:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=discriminator_hidden_states.hidden_states,
                attentions=discriminator_hidden_states.attentions,
            )
        else:
            return logits, cls_info

