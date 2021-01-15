# TSP-BGNN
Implementation of ["A bidirectional graph neural network for traveling salesman problems on arbitrary symmetric graphs"](https://www.sciencedirect.com/science/article/abs/pii/S0952197620303286), which is accepted at "Engineering Applications of Artificial Intelligence". If this code is useful for your work, please cite our paper:

    @article{hu97bidirectional,
      title={A bidirectional graph neural network for traveling salesman problems on arbitrary symmetric graphs},
      author={Hu, Yujiao and Zhang, Zhen and Yao, Yuan and Huyan, Xingpeng and Zhou, Xingshe and Lee, Wee Sun},
      journal={Engineering Applications of Artificial Intelligence},
      volume={97},
      pages={104061},
      publisher={Elsevier}
    }

## Dependencies
* Python >= 3.6
* Numpy
* PyTorch
* tqdm
* TensorboardX

## Usage
Before training the model, please first use gurobi/cplex or other tools that can compute optimal solution for  traveling salesman problems on arbitrary symmetric graphs, and then generate training dataset, validation dataset, testing dataset. 

Change the path of dataset in the codes.

Run the run.py and set the parameters.
