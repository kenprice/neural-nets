echo "TASK 1:  Neural network with only 1 hidden layer with 15 neurons."
python project.py 1 15 | grep Accuracy

echo "TASK 2:  Neural network with only 1 hidden layer with 150 neurons."
python project.py 1 150 | grep Accuracy

echo "TASK 3:  Neural network with 2 hidden layers with 100 neurons in first hidden layer and 15 neurons in the second hidden layer."
python project.py 2 100 15 | grep Accuracy

echo "TASK 4:  Neural network with 2 hidden layer with 500 neurons in first hidden layer and 150 neurons in the second hidden layer."
python project.py 2 500 150 | grep Accuracy
