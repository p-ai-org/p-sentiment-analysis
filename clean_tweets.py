import resources
import emoji #conda install emoji
import nltk #conda install nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def main():
    our_input = resources.get_input()
    # our_output = clean(our_input)
    third_party_input = resources.get_third_party_train()
    third_party_output = clean(third_party_input)

def dummy_text(tweet): #For testing. Replaces tweet text with a hardcoded string
    tweet.text = "ad 😀asdf☕️grw repeated repeated word the quick brown fox"
    return tweet

def convert_emoji(tweet): #Convert emoji to plaintext representation with underscores (e.g. "grinning_face")
    tweet.text = emoji.demojize(tweet.text, delimiters=[" ", " "]) #See https://pypi.org/project/emoji/
    return tweet

def underscores_to_spaces(tweet): #Replace underscores with spaces. Useful to apply on the output of convert_emoji, but not only that.
    tweet.text = tweet.text.replace("_", " ")
    return tweet

def remove_stop_words(tweet): #Remove stop words (common words to be ignored). Somewhat inefficient if the data will be re-tokenized later.
    tokens = word_tokenize(tweet.text) #See https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
    text = ""
    for token in tokens:
        if token not in stopwords.words("english"): text = text+" "+token
    tweet.text = text
    return tweet

def print_text(tweet): #Print the tweet's text.
    print(tweet.text)
    return tweet

def cleaning_step(tweet): #Template for a step in the cleaning pipeline (copypaste to add a new step)
    #Do something, probably with tweet.text
    return tweet

cleaning_pipeline = [
    # dummy_text,
    convert_emoji,
    underscores_to_spaces,
    remove_stop_words,
    # print_text
    ]

def clean(data):
    data = data.sample(n=40000, random_state=0) #Limit the data for now, for speed and ease of understanding
    for step in cleaning_pipeline:
        print("Applying "+step.__name__)
        data = data.apply(step, axis=1)
    return data
    

if __name__ == "__main__":
    main()