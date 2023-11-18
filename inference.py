import time
import stanza
import numpy as np
import concurrent.futures
from src.extract_all_features import Features
from src.lexical_diversity import LexicalDiversity
from src.sentence_complexity import SentenceComplexity


# Example usage
if __name__ == "__main__":
    # Assume para_list contains thousands of paragraphs
    sample_text_fiction = (
        "In the heart of the enchanted forest, where ancient trees whispered secrets and mystical creatures roamed freely, a peculiar phenomenon unfolded every full moon night. "
        "The silver moonbeams would weave a tapestry of shimmering light, revealing a hidden portal to a realm unknown. "
        "Amelia, a curious young girl with a penchant for adventure, discovered this ethereal gateway during a midnight stroll. "
        "Enticed by the allure of the unknown, she stepped through the luminous threshold, finding herself in a land bathed in hues of otherworldly colors. "
        "As she explored the fantastical realm, she encountered sentient plants that sang melodious tunes and elusive creatures that shimmered like stardust. "
        "Little did she know that her journey through the moonlit portal would unravel a destiny intertwined with the magic of the enchanted forest."
    )

    sample_text_non_fiction = (
        "In the heart of the bustling city, a vibrant market teemed with the sights and sounds of daily life. "
        "Local vendors, with their stalls overflowing with fresh produce and aromatic spices, created a kaleidoscope of colors and fragrances. "
        "The air buzzed with the energy of people from diverse walks of life, each engaged in the age-old tradition of buying and selling. "
        "Families navigated through the crowded lanes, sampling street food and negotiating prices. "
        "The market served not only as a commercial hub but also as a cultural melting pot, where the rich tapestry of the city's identity was woven through the exchange of goods and the sharing of stories."
    )
    para_list = [sample_text_fiction, sample_text_non_fiction]
    # para_list = []
    # for i in range(50):
    #     para_list.append(sample_text_fiction)
    #     para_list.append(sample_text_non_fiction)

    nlp = stanza.Pipeline(lang="en", processors={"tokenize": "spacy"})
    start_time = time.perf_counter()
    features = Features(para_list=para_list, nlp_pipeline=nlp)
    # Choose the choice: "raw", "lexical", "pos", or "syntactic"
    feature_values_parallel = features._extract_features(choice="syntactic")
    finish_time = time.perf_counter()
    print(f"Finished in {finish_time-start_time} seconds")

    # # Output the results
    # for i, para_features in enumerate(feature_values_parallel):
    #     print(f"Features for Paragraph {i + 1}: {para_features}")
