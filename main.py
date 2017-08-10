"""
It's mission
just read the parameters
then pass it to the agent then run it
"""
from params import *
from agent import Agent


def main():
    agent = Agent()
    agent.run()


if __name__ == '__main__':
    main()
