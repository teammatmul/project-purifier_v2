from transformers import ElectraTokenizer
import torch
import numpy as np

# from src import (
#     CONFIG_CLASSES,
#     TOKENIZER_CLASSES,
#     MODEL_FOR_SEQUENCE_CLASSIFICATION,
#     init_logger,
#     set_seed,
#     compute_metrics
# )

from processor import seq_cls_processors as processors
from processor import convert_single_example_to_feature

from modeling_electra import ElectraForSequenceClassification
# from puri_modeling import PurifierModel
# PurifierModel.from_pretrained("monologg/koelectra-small-discriminator")

model = ElectraForSequenceClassification.from_pretrained("./ckpt/checkpoint-11920/")
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-discriminator")

# GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

task = 'puri'
args = {}
processor = processors[task](args)
labels = processor.get_labels()

def puri_test_by_single_text(text, model=model, tokenizer=tokenizer):
    example = processor.create_single_example(text)
    feature = convert_single_example_to_feature(example, 128, tokenizer)

    with torch.no_grad():
        inputs = {
            "input_ids": torch.tensor(feature.input_ids, dtype=torch.long).unsqueeze(0).to(device),
            "attention_mask": torch.tensor(feature.attention_mask, dtype=torch.long).unsqueeze(0).to(device),
            "labels": None,
            "output_hidden_states": True
        }

        outputs = model(**inputs)

    preds = outputs[0].detach().cpu().numpy()

    return np.argmax(preds, axis=1), outputs

text = '안녕하세요 씨발'
out1, out2 = puri_test_by_single_text(text)
len(out2)

from modeling_bert import BertLayerNorm
LayerNorm = BertLayerNorm(256, eps=1e-12)

def select_layers(output_layers, selected_layers):
    # selected_layers is indices of index which 0 index means embedding_output
    mean_output_layers = output_layers[selected_layers[0]]
    if len(selected_layers) > 1:
        for idx in selected_layers[1:]:
            mean_output_layers = torch.add(mean_output_layers, output_layers[idx])
        mean_output_layers = LayerNorm(mean_output_layers)
    return mean_output_layers

len(out2[0])
len(out2)
len(out2[1])

sequence_output = out2[1]
query = [12]
query_hidden_states = select_layers(sequence_output, query)
query_hidden_states.shape

from configuration_utils import PretrainedConfig
PretrainedConfig.from_json_file('./config/config.json')

out1
len(out2)

out2[0].shape

len(out2[1])
out2[1][0].shape

len(out2[2])
out2[2][0].shape
len(out2)

import pandas as pd
df_eval = pd.read_csv('./data/puri_test_set.csv')
df_eval['predict'] = df_eval.text.apply(lambda x: puri_test_by_single_text(str(x)))
df_eval['predict'] = df_eval.predict.apply(lambda x: x[0])
df_eval.to_csv('./data/eval_result/electra_10epoch_test.csv', index=False)