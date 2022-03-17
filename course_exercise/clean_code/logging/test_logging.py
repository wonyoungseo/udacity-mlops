## STEPS TO COMPLETE ##
# 1. import logging
# 2. set up config file for logging called `test_results.log`
# 3. add try except with logging and assert tests for each function
#    - consider denominator not zero (divide_vals)
#    - consider that values must be floats (divide_vals)
#    - consider text must be string (num_words)
# 4. check to see that the log is created and populated correctly
#    should have error for first function and success for
#    the second
# 5. use pylint and autopep8 to make changes
#    and receive 10/10 pep8 score

'''
Logging & testing exercise (in progress)

author: wyseo
date: 2021-09-09
'''

# import logging
import logging

# set up config file for logging called `test_results.log`
logging.basicConfig(
    filename='./test_results.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
)



def divide_vals(numerator, denominator):
    '''
    Args:
        numerator: (float) numerator of fraction
        denominator: (float) denominator of fraction

    Returns:
        fraction_val: (float) numerator/denominator
    '''

    # add try except with logging and assert tests
    try:
        # consider denominator not zero(divide_vals)
        assert denominator != 0

        # consider that values must be floats (divide_vals)
        assert isinstance(numerator, float)
        assert isinstance(denominator, float)


        fraction_val = numerator/denominator
        logging.info("Division SUCCESS")
        return fraction_val
    except AssertionError:
        logging.error("Division FAILED: type needs to be float and denominator cannot be zero.")


def num_words(text):
    '''
    Args:
        text: (string) string of words

    Returns:
        num_words: (int) number of words in the string
    '''
    # add try except with logging and assert tests
    try:
        # consider text must be string (num_words)
        assert isinstance(text, str)
        num_words = len(text.split())
        logging.info("Counting number of words: SUCCESS")
        return num_words
    except AssertionError:
        logging.error("Counting number of words FAILED: The type needs to be a string")

if __name__ == "__main__":
    divide_vals(3.4, 0)
    divide_vals(4.5, 2.7)
    divide_vals(-3.8, 2.1)
    divide_vals(1, 2)
    num_words(5)
    num_words('This is the best string')
    num_words('one')
