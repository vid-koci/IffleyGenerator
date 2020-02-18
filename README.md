# IffleyGenerator
NN-based generator of climbing routes at Iffley gym in Oxford.

To run the code, following libraries are needed:
  * Python3.6 or higher
  * Pytorch 1.0 or higher
  * tqdm is used to display progress during training
  * Cuda is recommended to speed up the training

To generate some routes, run the `python generate.py` file. It will generate routes and write them into the file named `generated.txt`.
Run `python generate.py -h` to get information on additional options.

To re-run the training, see `main.py` file. Best hyperparameters found were set as defaults.
