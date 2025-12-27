import torch

def collate_fn(batch, tokenizer, seq_len: int, is_decoder_only: bool = False) -> dict:
    input_ids = []
    attention_masks = []
    for sample in batch:
        input_ids.append(sample["input_ids"])
        attention_masks.append(sample["attention_mask"])
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.tokenizer.pad_token_id
    )
    attention_masks = torch.nn.utils.rnn.pad_sequence(
        attention_masks,
        batch_first=True,
        padding_value=0
    )
    input_ids = input_ids[:, :seq_len]
    attention_masks = attention_masks[:, :seq_len]

    if is_decoder_only:
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels
        }

    else:
        decoder_input_ids = input_ids[:, :-1]
        labels = input_ids[:, 1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels
        }
