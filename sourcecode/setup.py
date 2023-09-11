import sys
import os

def setup():
    print('SETUP PATH-VARIABLES...')
    ROOT_DIR = os.path.abspath(os.curdir)
    HELPERS_DIR = ROOT_DIR + "/sourcecode/helpers"
    WEBSCRAPER_DIR = ROOT_DIR + "/sourcecode/webscraper"
    NEEDLEMAN_WUNSCH_DIR = ROOT_DIR + "/sourcecode/needleman_wunsch"

    sys.path.append(HELPERS_DIR)
    sys.path.append(WEBSCRAPER_DIR)
    sys.path.append(NEEDLEMAN_WUNSCH_DIR)