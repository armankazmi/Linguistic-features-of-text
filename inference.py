import time
import stanza
import numpy as np
import pandas as pd
from src.extract_all_features import Features


# Example usage
if __name__ == "__main__":
    sample_text_fiction = (
        "In the heart of the enchanted forest, where ancient trees whispered secrets and mystical creatures roamed freely, a peculiar phenomenon unfolded every full moon night. "
        "The silver moonbeams would weave a tapestry of shimmering light, revealing a hidden portal to a realm unknown. "
        "Amelia, a curious young girl with a penchant for adventure, discovered this ethereal gateway during a midnight stroll. "
        "Enticed by the allure of the unknown, she stepped through the luminous threshold, finding herself in a land bathed in hues of otherworldly colors. "
        "As she explored the fantastical realm, she encountered sentient plants that sang melodious tunes and elusive creatures that shimmered like stardust. "
        "Little did she know that her journey through the moonlit portal would unravel a destiny intertwined with the magic of the enchanted forest."
    )
    sample_text_fiction_2 = (
        "In a groundbreaking oceanic expedition, scientists announced the discovery of an ancient, submerged city beneath the waves. "
        "The city, estimated to be thousands of years old, revealed intricate architecture and mysterious artifacts that have left researchers in awe. "
        "Initial findings suggest a highly advanced civilization once thrived in the deep, challenging conventional timelines of human history. "
        "Archaeologists and marine biologists joined forces for this unprecedented venture, utilizing state-of-the-art submersibles to explore the city's ruins. "
        "Experts are already planning additional expeditions to delve deeper into Atlantia's mysteries, hoping to uncover the secrets of its inhabitants and the events that led to its submersion. "
        "The discovery has ignited the imaginations of scientists and the public alike, sparking discussions about the implications of such a find on our understanding of ancient civilizations and the mysteries that lie hidden beneath the Earth's oceans."
    )

    sample_text_non_fiction = (
        "In the heart of the bustling city, a vibrant market teemed with the sights and sounds of daily life. "
        "Local vendors, with their stalls overflowing with fresh produce and aromatic spices, created a kaleidoscope of colors and fragrances. "
        "The air buzzed with the energy of people from diverse walks of life, each engaged in the age-old tradition of buying and selling. "
        "Families navigated through the crowded lanes, sampling street food and negotiating prices. "
        "The market served not only as a commercial hub but also as a cultural melting pot, where the rich tapestry of the city's identity was woven through the exchange of goods and the sharing of stories."
    )
    sample_text_non_fiction_2 = (
        "A community is a social unit (a group of living things) with a shared socially significant characteristic, such as place, set of norms, culture, religion, values, customs, or identity. "
        "Communities may share a sense of place situated in a given geographical area (e.g. a country, village, town, or neighbourhood) or in virtual space through communication platforms. "
        "Durable good relations that extend beyond immediate genealogical ties also define a sense of community, important to their identity, practice, and roles in social institutions such as family, home, work, government, TV network, society, or humanity at large. "
        "Although communities are usually small relative to personal social ties, 'community' may also refer to large group affiliations such as national communities, international communities, and virtual communities. "
    )

    sample_text_mixed = (
        "In a small coastal town, nestled between rolling hills and the vast expanse of the ocean, there was a legendary lighthouse with a rich history. "
        "Local folklore spoke of a ghostly figure, the keeper of the light, who was said to appear during storms, guiding lost ships safely to shore. "
        "Despite the tales, the lighthouse stood as a steadfast beacon, providing a very real source of navigation for sailors navigating the treacherous waters. "
        "Adjacent to this maritime folklore, a team of marine biologists embarked on a real-world expedition to study the diverse ecosystems thriving beneath the ocean's surface. "
        "Equipped with cutting-edge technology, they explored the depths, discovering new species and documenting the intricate balance of marine life. "
        "The intersection of these narratives, where maritime legends met the empirical pursuits of marine science, underscored the fascinating blend of myth and reality that defined the coastal community's identity. "
    )

    para_list = [
        sample_text_fiction,
        sample_text_fiction_2,
        sample_text_non_fiction,
        sample_text_non_fiction_2,
        sample_text_mixed,
    ]

    ## Feature calculation
    nlp = stanza.Pipeline(lang="en", processors={"tokenize": "spacy"})
    start_time = time.perf_counter()

    features = Features(para_list=para_list, nlp_pipeline=nlp)
    # Choice = "all" for calculating all features. Other options available are: 'raw', 'lexical', 'syntactic' and 'pos'
    feature_values = features._extract_features(choice="all")

    finish_time = time.perf_counter()
    print(f"Finished in {finish_time-start_time} seconds")

    # Making the predictions
    import pickle

    with open("resources/final_model.pkl", "rb") as fp:
        model = pickle.load(fp)

    # best 28 features
    best_feat = [
        "TTR",
        "Maas TTR",
        "VocD",
        "adverb/pronoun",
        "noun/verb",
        "mark",
        "nsubj",
        "nummod",
        "acl:relcl",
        "nmod:poss",
        "flat",
        "fixed",
        "aux:pass",
        "obl:npmod",
        "discourse",
        "('VERB', 'ADV', 'before')",
        "('VERB', 'PROPN', 'after')",
        "('VERB', 'ADP', 'before')",
        "('ADJ', 'SCONJ', 'after')",
        "('VERB', 'PRON', 'before')",
        "('VERB', 'SCONJ', 'after')",
        "('PRON', 'VERB', 'before')",
        "('PRON', 'NOUN', 'before')",
        "('PROPN', 'NUM', 'before')",
        "('PROPN', 'PROPN', 'after')",
        "('VERB', 'NUM', 'before')",
        "std_sen_len",
        "content/function",
    ]

    for para_features in feature_values:
        dependency_features = {
            i: j
            for k, v in para_features.items()
            if k == "dependency features"
            for i, j in v.items()
        }
        other_features = {
            i: j for i, j in para_features.items() if i != "dependency features"
        }

        updated_feats = {**dependency_features, **other_features}

        selected_feats = {feat: updated_feats.get(feat, 0) for feat in best_feat}

        x = pd.DataFrame([selected_feats])
        fiction_probablity = model.predict_proba(x)[0][1]
        tag = "fiction" if fiction_probablity > 0.5 else "non-fiction"
        prob = fiction_probablity if tag == "fiction" else 1 - fiction_probablity

        print(
            "The given paragraph is ",
            tag,
            " with a probabilty of {:.3f}".format(prob * 100),
            "%",
        )
