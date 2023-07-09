python37 -m poetry shell
git clone -b rl_projects https://github.com/rdaluiso/baselines.git
cd ./baselines
pip install -e .
cd ..
\rm -r ./baselines