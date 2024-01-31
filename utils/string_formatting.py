import pandas as pd

def remove_newline(text):
    return ' '.join(text.splitlines())

def remove_multiple_whitespace(text):
    return " ".join(text.split())

def format_claim(text):
    formated_text = remove_multiple_whitespace(remove_newline(text))
    return formated_text

def remove_char(s, c) :
     
    # find total no. of
    # occurrence of character
    counts = s.count(c)
 
    # convert into list
    # of characters
    s = list(s)
 
    # keep looping until
    # counts become 0
    while counts :
         
        # remove character
        # from the list
        s.remove(c)
 
        # decremented by one
        counts -= 1
 
    # join all remaining characters
    # of the list with empty string
    s = '' . join(s)
     
    return s