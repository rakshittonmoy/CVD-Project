# CS59300CVD Assignment 1

## Create the virtual environment:-

conda create -n alignenv python=3.12.5 
conda activate alignenv  

## Install the dependencies using:- 

conda install --file config/requirements.txt

## Update the requirements file each time a new package is installed in the conda env using:-

conda env export > environment.yml

## Assignment 1:- Run the code using:-

python3 main_p1.py -i data/6.jpg -m mse

python3 main_p1.py -i all

## Assignment 2:- Run the code using:-

python3 main_p2.py -i all -s 1.8 -k 25 -n 15

## Assignment 3:- Run the code using:-

python3 main.py
