import random

def mask_tokens(text, mask_token = "[MASK]", p = 0.05):
    words = text.split()
    new_words = []
    for w in words:
        if random.random() < p:
            new_words.append(mask_token)
        else:
            new_words.append(w)
    return " ".join(new_words)