"""
This module is the entrypoint for the project. It will handle the order of
task execution of training and analysing models.

Last updated: MB 12/08/2020 - created module
"""
# import local modules.
from model import basic_regression

def main():
    basic_regression.regression_2019()

if __name__ == '__main__':
    main()
