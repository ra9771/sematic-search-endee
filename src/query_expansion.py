"""
Query Expansion
Enriches the user's query with synonyms and related terms using WordNet (NLTK).
This improves recall — especially for TF-IDF — by bridging vocabulary gaps.
"""

import logging
import re
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

# Download required NLTK data on first use
def _ensure_nltk():
    for resource in ["wordnet", "punkt", "stopwords", "averaged_perceptron_tagger", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt") else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


class QueryExpander:
    """
    Expands a query by appending WordNet synonyms for key terms.
    Helps surface relevant documents that use different vocabulary.
    """

    def __init__(self, max_synonyms_per_word: int = 2, max_expanded_terms: int = 5):
        _ensure_nltk()
        self.max_synonyms_per_word = max_synonyms_per_word
        self.max_expanded_terms = max_expanded_terms
        self.stop_words = set(stopwords.words("english"))

    def _get_synonyms(self, word: str) -> list[str]:
        """Return top synonyms for a word from WordNet."""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ").lower()
                if name != word.lower() and name not in self.stop_words:
                    synonyms.add(name)
                if len(synonyms) >= self.max_synonyms_per_word:
                    break
            if len(synonyms) >= self.max_synonyms_per_word:
                break
        return list(synonyms)

    def expand(self, query: str) -> str:
        """
        Expand a query with synonyms.

        Args:
            query: Original user query string.

        Returns:
            Expanded query string with synonyms appended.
        """
        tokens = word_tokenize(query.lower())
        # Keep only meaningful content words
        keywords = [t for t in tokens if t.isalpha() and t not in self.stop_words]

        added_terms = []
        for word in keywords:
            if len(added_terms) >= self.max_expanded_terms:
                break
            synonyms = self._get_synonyms(word)
            for syn in synonyms:
                if syn not in query.lower() and syn not in added_terms:
                    added_terms.append(syn)

        if added_terms:
            expanded = query + " " + " ".join(added_terms)
            logger.debug(f"Query expanded: '{query}' → '{expanded}'")
            return expanded

        return query

    def expand_list(self, queries: list[str]) -> list[str]:
        """Expand a list of queries."""
        return [self.expand(q) for q in queries]
