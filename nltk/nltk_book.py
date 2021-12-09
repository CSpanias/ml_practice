from nltk.book import *

# concordance view: every occurrence of a given word along with context
print(text1.concordance("whale"))
# other words that appear in a similar range of contexts
print(text1.similar("monstrous"))
# examine just the contexts that are shared by 2 or more words
print(text2.common_contexts(['monstrous', 'very']))
# dispersion plot: positional information of a word
print(text4.dispersion_plot(["citizens", "freedom"]))
# generate random text in the same style
print(text3.generate())
# find the length of a text (words, punctuation)
print(len(text3))
# obtain the vocabulary (distinct words) of a text
print(sorted(set(text3)))
# count the distinct words
print(len(set(text3)))
# calculate the lexical richness
print(len(set(text3)) / len(text3))
# frequency of a word
print(text3.count("smote"))
# word frequency in %
print(100 * text4.count('a') / len(text4))


def lexical_diversity(text):
    """Calculate how many unique words (including punctuation) are in
    a text file."""
    print("The lexical diversity of", text, "is",
          round(len(set(text)) / len(text), 2), ".")


def percentage(count, total):
    """Calculate how many unique words (including punctuation) are in
    a text file in relative terms."""
    print("The % of lexical diversity is:", round(100 * count / total, 2), ".")


lexical_diversity(text5)
percentage(text5.count('hello'), len(text4))
