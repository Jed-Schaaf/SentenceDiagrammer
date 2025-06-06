def check_input(sentence):
    """
    Validate the input sentence.
    Returns an error message if invalid, None otherwise.
    """
    if not sentence or len(sentence.strip()) == 0:
        return "Please enter a valid sentence."
    return None