## STEPS TO COMPLETE ##
# 1. import logging
# 2. set up config file for logging called `results.log`
# 3. add try except with logging for success or error
#    in relation to checking the types of a and b
# 4. check to see that log is created and populated correctly
#    should have error for first function and success for
#    the second
# 5. use pylint and autopep8 to make changes
#    and receive 10/10 pep8 score

'''
Logging exercise (now solved)

author: wyseo
date: 2021-09-09
'''

import logging

logging.basicConfig(
    filename='./results.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
)


def sum_vals(val_1: int, val_2: int) -> int:
    """
    Sum the input values
    :param val_1: (int)
    :param val_2: (int)
    :return: val_1 + val_2 (int)
    """
    try:
        assert isinstance(val_1, int)
        assert isinstance(val_2, int)
    except AssertionError:
        logging.info(
            "Input Type: val_1 - %s // val_2 - %s",
            type(val_1),
            type(val_2))
        logging.error("ERROR: Both input val_1, val_2 must be integer.")
        return None
    else:
        logging.info("SUCCESS: Summing input values")
        return val_1 + val_2


if __name__ == "__main__":
    sum_vals('no', 'way')
    sum_vals(4, 5)
