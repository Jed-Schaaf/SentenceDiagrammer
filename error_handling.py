def check_input(text):
    """
    Validate the input sentence.
    Returns an error message if invalid, None otherwise.
    """
    if not text or len(text.strip()) == 0:
        return "Please enter at least one valid sentence."
    return None