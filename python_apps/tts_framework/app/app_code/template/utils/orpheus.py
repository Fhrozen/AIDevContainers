import numpy as np
import torch


def parse_output(generated_ids):
    token_to_find = 128257
    token_to_remove = 128258

    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()
        cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
    else:
        cropped_tensor = generated_ids

    processed_rows = []
    for row in cropped_tensor:
        masked_row = row[row != token_to_remove]
        processed_rows.append(masked_row)

    code_lists = []
    for row in processed_rows:
        row_length = row.size(0)
        new_length = (row_length // 7) * 7
        trimmed_row = row[:new_length]
        trimmed_row = [t - 128266 for t in trimmed_row]
        code_lists.append(trimmed_row)
    codes = torch.tensor(code_lists[0]).cpu().numpy()  # Return just the first one for single sample
    return codes


# Redistribute codes for audio generation
def redistribute_codes(code_list):
    layer_1 = []
    layer_2 = []
    layer_3 = []
    for i in range((len(code_list)+1)//7):
        layer_1.append(code_list[7*i])
        layer_2.append(code_list[7*i+1]-4096)
        layer_3.append(code_list[7*i+2]-(2*4096))
        layer_3.append(code_list[7*i+3]-(3*4096))
        layer_2.append(code_list[7*i+4]-(4*4096))
        layer_3.append(code_list[7*i+5]-(5*4096))
        layer_3.append(code_list[7*i+6]-(6*4096))
        
    layer_1 = np.array(layer_1)[None]
    layer_2 = np.array(layer_2)[None]
    layer_3 = np.array(layer_3)[None]
    return {
        "audio_codes.0": layer_1,
        "audio_codes.1": layer_2,
        "audio_codes.2": layer_3,
    }
