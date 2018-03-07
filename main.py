"""
It's mission
just read the parameters and create the dirs for the exp
then pass it to the agent then run it
"""

from agent import Agent
from utils.params import get_params
from utils.dirs import create_exp_dirs

def main():
    args = get_params()
    args = create_exp_dirs(args)
    agent = Agent(args)
    agent.run()


if __name__ == '__main__':
    main()
