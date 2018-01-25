conda create -n deepqa_env python=3.6 pytorch torchvision matplotlib -c pytorch

source activate deepqa_env

mkdir DeepQA
cd DeepQA
git clone https://github.com/pwieler/DeepQA.git .

mkdir data
cd data
wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
tar -xzf tasks_1-20_v1-2.tar.gz

cd ..
