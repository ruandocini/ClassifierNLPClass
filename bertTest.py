from transformers import AutoModel
from transformers import AutoTokenizer
import torch
import numpy as np

model = AutoModel.from_pretrained("neuralmind/bert-large-portuguese-cased")
tokenizer = AutoTokenizer.from_pretrained(
    "neuralmind/bert-large-portuguese-cased", do_lower_case=False
)


sentences = [
    "Tinha uma pedra no meio do caminho.",
    "Eu gosto de cachorro",
    "Eu gosto de gato",
    "Eu gosto de passarinho",
]

input_ids = [
    tokenizer.encode(sentence, return_tensors="pt") for sentence in sentences
]

with torch.no_grad():
    outs = [model(input_id) for input_id in input_ids]
    encoded = [out[0][0, 1:-1] for out in outs]
    final_representation = np.array(
        [np.mean(sent.cpu().detach().numpy(), axis=0) for sent in encoded]
    )
    print(final_representation.shape)
