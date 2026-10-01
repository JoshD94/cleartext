from pos_tag import pos_tag

tokens = pos_tag("The committee utilized innovative methodologies.")
for t in tokens:
    if t["pos"] in ("NOUN", "VERB", "ADJ", "ADV"):
        print(t["text"], t["pos"], t["tag"])