import re
from tqdm import tqdm
from unidecode import unidecode
from textacy.preprocessing.normalize import normalize_unicode
from textacy.preprocessing.replace import replace_urls, replace_emails, \
    replace_phone_numbers, replace_numbers, replace_currency_symbols
from textacy.preprocessing.remove import remove_accents, remove_punctuation
from keras_preprocessing.text import text_to_word_sequence

tqdm.pandas()

LINEBREAK_REGEX = re.compile(r'((\r\n)|[\n\v])+')
NONBREAKING_SPACE_REGEX = re.compile(r'(?!\n)\s+')

def normalize_whitespace(text):
    return NONBREAKING_SPACE_REGEX.sub(' ', LINEBREAK_REGEX.sub(r'\n', text)).strip()

def unpack_contractions(text):
    # standard
    text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|"
                  r"[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't",
                  r"\1\2 not", text)
    text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll",
                  r"\1\2 will", text)
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
    text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve",
                  r"\1\2 have", text)
    # non-standard
    text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
    text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
    text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
    text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
    text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
    text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)
    return text

def default_cleaner(text, fix_unicode=True, lowercase=True, transliterate=True,
                    no_urls=True, no_emails=True, no_phone_numbers=True,
                    no_numbers=True, no_currency_symbols=True, no_punct=True,
                    no_contractions=False, no_accents=True):
    if fix_unicode:
        text = normalize_unicode(text, form='NFC')
    if transliterate is True:
        text = unidecode(text)
    if lowercase is True:
        text = text.lower()
    if no_urls:
        text = replace_urls(text, '<URL>')
    if no_emails is True:
        text = replace_emails(text, '<EMAIL>')
    if no_phone_numbers is True:
        text = replace_phone_numbers(text, '<PHONE>')
    if no_numbers is True:
        text = replace_numbers(text, '<NUMBER>')
    if no_currency_symbols is True:
        text = replace_currency_symbols(text, '<CUR>')
    if no_contractions is True:
        text = unpack_contractions(text)
    if no_accents is True:
        text = remove_accents(text)
    if no_punct is True:
        text = remove_punctuation(text)
    return normalize_whitespace(text)

def default_tokenizer(text):
    return text_to_word_sequence(text, filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n', lower=False)

def ktext_process_text(series, cleaner=default_cleaner,
                       tokenizer=default_tokenizer,
                       append_indicators=False):
    if append_indicators:
        return series.progress_map(lambda x: ['<SOS>'] + tokenizer(cleaner(x)) + ['<EOS>'])
    return series.progress_map(lambda x: tokenizer(cleaner(x)))
