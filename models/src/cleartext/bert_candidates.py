"""Experimental masked-BERT proposals. These are not accepted simplifications.

Novel proposals have no invented WordNet senses. The current relation decision
and meaning floor require a separate evaluation before this generator is routed
into a deployed pipeline.
"""

from lemminflect import getInflection, getLemma

from . import contextual
from .complete_slot import slot_inputs
from .generation import (
    CandidateGenerator,
    TAGS,
    hyphen_fragment,
    relation_map,
    resolve_target,
)


def paired_input(tokenizer, context, start, end, limit=512):
    segment_limit = (limit + 1) // 2
    original = slot_inputs(
        tokenizer, context, start, end, [context[start:end]], limit=segment_limit
    )[0]
    masked = slot_inputs(
        tokenizer, context, start, end, [tokenizer.mask_token], limit=segment_limit
    )[0]
    ids = original.input_ids + masked.input_ids[1:]
    types = (0,) * len(original.input_ids) + (1,) * (len(masked.input_ids) - 1)
    position = len(original.input_ids) + masked.positions[0] - 1
    return ids, types, position


def masked_predictions(requests, top_k=20, batch_size=16):
    import torch

    if top_k < 1 or batch_size < 1:
        raise ValueError("top_k and batch_size must be positive")
    if not requests:
        return []
    tokenizer, model = contextual.model()
    inputs = [
        paired_input(tokenizer, *request, limit=model.config.max_position_embeddings)
        for request in requests
    ]
    result = []
    with torch.no_grad():
        for offset in range(0, len(inputs), batch_size):
            chunk = inputs[offset : offset + batch_size]
            length = max(len(ids) for ids, _, _ in chunk)
            ids = torch.tensor(
                [
                    list(ids) + [tokenizer.pad_token_id] * (length - len(ids))
                    for ids, _, _ in chunk
                ]
            )
            types = torch.tensor(
                [list(types) + [0] * (length - len(types)) for _, types, _ in chunk]
            )
            mask = torch.tensor(
                [[1] * len(ids) + [0] * (length - len(ids)) for ids, _, _ in chunk]
            )
            hidden = model.bert(
                input_ids=ids, attention_mask=mask, token_type_ids=types
            ).last_hidden_state
            positions = torch.tensor([position for _, _, position in chunk])
            probabilities = model.cls(
                hidden[torch.arange(len(chunk)), positions]
            ).softmax(-1)
            values, indices = probabilities.topk(
                min(top_k, probabilities.shape[1]), dim=-1
            )
            for scores, tokens in zip(values.tolist(), indices.tolist()):
                result.append(
                    [
                        {
                            "token": tokenizer.convert_ids_to_tokens(token),
                            "probability": score,
                            "rank": rank + 1,
                        }
                        for rank, (token, score) in enumerate(zip(tokens, scores))
                    ]
                )
    return result


def proposals(token, predictions, target=None):
    if token is None or hyphen_fragment(token):
        return None, []
    target = resolve_target(token) if target is None else target
    if target is None or target[2] not in TAGS.get(target[1], ()):
        return target, []
    lemma, pos, tag, fallback = target
    origins = {
        name.lower(): senses for name, (_, senses) in relation_map(lemma, pos).items()
    }
    upos = {"n": "NOUN", "v": "VERB", "a": "ADJ", "r": "ADV"}[pos]
    found = {}
    for prediction in predictions:
        surface = prediction["token"]
        if not surface.isalpha():
            continue
        for base in getLemma(surface, upos=upos, lemmatize_oov=False):
            forms = getInflection(base, tag=tag, inflect_oov=False)
            if not forms:
                continue
            word = forms[0]
            if word.lower() == token.text.lower():
                continue
            if token.text.isupper():
                word = word.upper()
            elif token.text.istitle():
                word = word[:1].upper() + word[1:]
            found.setdefault(
                word.lower(),
                {
                    "word": word,
                    "lemma": base,
                    "source": "bert",
                    "senses": list(origins.get(base.lower(), ())),
                    "multiword": False,
                    "pos_fallback": fallback,
                    "bert_token": surface,
                    "bert_rank": prediction["rank"],
                    "bert_probability": prediction["probability"],
                },
            )
    return target, list(found.values())


class MaskedBertGenerator(CandidateGenerator):
    name = "masked_bert_proposals"

    def __init__(self, top_k=20):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.top_k = top_k

    def generate(self, token):
        if token is None or hyphen_fragment(token):
            return None, []
        target = resolve_target(token)
        if target is None:
            return target, []
        predictions = masked_predictions(
            [(token.doc.text, token.idx, token.idx + len(token))], self.top_k
        )[0]
        return proposals(token, predictions, target)
