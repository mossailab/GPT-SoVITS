
from transformers import BertTokenizer, BertModel
import torch
from GPT_SoVITS.text.cleaner import clean_text


def build_text_tokens_and_features(text, version, prompt_text, device):
    """
    跳过 G2P/NLTK，直接构造 phones 和 BERT 特征
    """
    phones, word2ph, norm_text = clean_text(text, "zh", version)

    tokenizer = BertTokenizer.from_pretrained("GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
    model = BertModel.from_pretrained("GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large").to(device)
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(prompt_text + text, return_tensors="pt").to(device)
        outputs = model(**inputs)
        bert_features = outputs.last_hidden_state[0].cpu().T  # shape: (1024, seq_len)

    return phones, bert_features, norm_text
