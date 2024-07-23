import re
from enum import Enum


class BinaryPatterns(Enum):
    """Enum class for binary patterns. We can add more patterns than just for matchin yes/no answers and the pattern can be extended."""
    YES = re.compile(r'\b(yes|yeah|yep)\b', re.IGNORECASE)
    NO = re.compile(r'\b(no|nah|nope)\b', re.IGNORECASE)


class BinaryMapping:
    """This class handles the mapping of textual answers to binary values like 1 or 0"""

    def __init__(self):
        # Initialize patterns based on the enum values
        self.patterns = {
            BinaryPatterns.YES: 1,
            BinaryPatterns.NO: 0,
            # Add more patterns and associated values as needed
        }

    def map_to_binary(self, answer):
        """
       Maps a textual answer to a binary value based on predefined patterns. Example usage:
       binary_mapping.map_to_binary("Yes, that's correct.") would return 1.
       Args: answer (str): The textual answer to be mapped to a binary value.
       Returns: int or None: The binary value (0 or 1) if a matching pattern is found; otherwise, None.
       """
        matched_answer = None
        for pattern, value in self.patterns.items():
            match = pattern.value.search(answer)
            if match:
                matched_answer = value
                return matched_answer
        return matched_answer


def clean_answer(answer):
    # Remove newlines and tabs using regex
    # add more code to clean the answers from the models, e.g. eventually we have to remove the repeated prompt etc.
    cleaned_answer = re.sub(r'[\n\t]+', ' ', answer)
    return cleaned_answer


