from transformers import ElectraTokenizer
import torch
import numpy as np

from processor import seq_cls_processors as processors
from processor import convert_single_example_to_feature

from puri_modeling import PurifierModel
model = PurifierModel.from_pretrained("./ckpt/checkpoint-12820/")

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-discriminator")

# GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

task = 'puri'
args = {}
processor = processors[task](args)
labels = processor.get_labels()

def masking_sentence_well(tokens, toxic_ids):
    toxic_ids = toxic_ids - 1
    drop_list = [toxic_ids]
    if len(tokens[toxic_ids]) >= 3:
        # front
        for tok in tokens[:toxic_ids][::-1]:
            if tok[:2] == '##':
                drop_list.append(tokens.index(tok))
            else:
                drop_list.append(tokens.index(tok))
                break

    for tok in tokens[toxic_ids + 1:]:
        if tok[:2] == '##':
            drop_list.append(tokens.index(tok))
        else:
            break

    drop_list.sort()

    return tokens[:drop_list[0]] + ['*'] + tokens[drop_list[-1] + 1:]

def single_purifier_process(text, model=model, tokenizer=tokenizer):
    example = processor.create_single_example(text)
    feature = convert_single_example_to_feature(example, 128, tokenizer)

    with torch.no_grad():
        inputs = {
            "input_ids": torch.tensor(feature.input_ids, dtype=torch.long).unsqueeze(0).to(device),
            "attention_mask": torch.tensor(feature.attention_mask, dtype=torch.long).unsqueeze(0).to(device),
            "labels": None,
            "output_hidden_states": True,
            "query": [12],
            "key": [1, 2, 3],
            "value": [1, 2, 3],
            "query_att": True,
            "key_att": True,
            "multi_head": True
        }

        outputs, cls_info = model(**inputs)

    preds = outputs[0].detach().cpu().numpy()
    result = np.argmax(preds, axis=0)
    return result, cls_info


def full_purifier_process(text, model=model, tokenizer=tokenizer):
    result = 1
    for_result = 1
    while result:
        example = processor.create_single_example(text)
        feature = convert_single_example_to_feature(example, 128, tokenizer)

        with torch.no_grad():
            inputs = {
                "input_ids": torch.tensor(feature.input_ids, dtype=torch.long).unsqueeze(0).to(device),
                "attention_mask": torch.tensor(feature.attention_mask, dtype=torch.long).unsqueeze(0).to(device),
                "labels": None,
                "output_hidden_states": True,
                "query": [12],
                "key": [1, 2, 3],
                "value": [1, 2, 3],
                "query_att": True,
                "key_att": True,
                "multi_head": True
            }

            outputs, cls_info = model(**inputs)

        preds = outputs[0].detach().cpu().numpy()
        result = np.argmax(preds, axis=0)

        if for_result:
            final_result = result
            for_result = 0

        if result == 0:
            return text, final_result

        tokens = tokenizer.tokenize(text)

        toxic_ids = list(cls_info['probs'][0][0][0]).index(max(cls_info['probs'][0][0][0]))
        masked_tokens = masking_sentence_well(tokens, toxic_ids)
        if text == tokenizer.convert_tokens_to_string(masked_tokens):
            return text, final_result

        text = tokenizer.convert_tokens_to_string(masked_tokens)

text = "니 대가리 속은 우동사리로 채워져있냐?"
text, result = full_purifier_process(text)

# 후처리
text = text.replace(" ?","?").replace(" !","!").replace(" .",".").replace(" ,",",")













