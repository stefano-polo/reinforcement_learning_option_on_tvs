import sys

import envs.fe_envs  # import the object enviroment of my problem
from baselines.run import (
    main,
)  # import the main function of run.py of baselines library

if __name__ == "__main__":
    main(sys.argv)
