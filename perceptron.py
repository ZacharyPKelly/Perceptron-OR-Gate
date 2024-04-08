#######################################################################################################################################
# This file contains code in order to implement the perceptron model for AUCSC 460 Lab Assignment Four
#
# Class: AUCSC 460
# Name: Zachary Kelly
# Student ID: 1236421
# Date: March 6th, 2024
#######################################################################################################################################

# IMPORTS #
import numpy as np

# PRECEPTRON CLASS AND ASSOCIATED METHODS #

class Perceptron():

    def __init__(self, learningRate = 0.05, iterations = 1000):

        """
        Initializes the variables to be used in the Perceptron class.

        Iterations and learning rate can be adjusted here.
        """

        #Set iterations
        self.iterations = iterations

        #set learningRate
        self.learningRate = learningRate

        #Initialize bias variable
        self.bias = None

        #initialize weights variable
        self.weights = None

    def stepFunction(self, weightedSum):

        """
        Determines based on the weighted sum of the test example, weights, and bias what the prediction of the model will be
        weightedSum: The weighted sum of the test example, weights, and bias

        Returns 1 if weightedSum is greater than or equal to 0, 0 otherwise
        """

        if weightedSum < 0:

            return 0
        
        else:

            return 1
        
    def fit(self, trainExamples, trainTargets):

        """
        Fits the perceptron model to the given data. Modifies the weights and bias in order to correctly fit the model.
        trainExamples : The training dataset as an array filled with OR Gate examples as 1d arrays. [[a,b], [c,d]...]
        trainTargets  : The correct outputs for the trainExamples dataset
        """

        numOfExamples, numOfFeatures = trainExamples.shape

        #initializing our weights and bias randomly
        self.weights = np.random.uniform(size = numOfFeatures, low = -0.5, high = 0.5)
        self.bias = np.random.uniform(low = -0.5, high = 0.5)

        print("-----------------------INFO-----------------------\n")
        print("Starting Weights : ", self.weights)
        print("Starting Bias    : ", self.bias)
        print("Learning Rate    : ", self.learningRate)
        print("Iterations       : ", self.iterations)
        print("\n--------------------------------------------------")

        for x in range(self.iterations):

            for correctOutput, selectedExample in zip(trainTargets, trainExamples):
                
                #weightedSum is the dot multiplication of weights and example plus the bias
                weightedSum = np.dot(selectedExample, self.weights) + self.bias

                #Run our weighted sum through the step function to determine the prediction
                predictedOutput = self.stepFunction(weightedSum)

                #Update Weights based on the correctness of our prediction
                self.updateWeights(selectedExample, correctOutput, predictedOutput)
    
    def updateWeights(self, selectedExample, correctOutput, predicatedOutput):

        """
        Updates the weights for the perceptron model.
        selectedExample : The given OR Gate example as a 1d array [x, y]
        correctOutput   : The correct output of the given OR Gate example
        predictedOutput : The predicted output by the model of the given OR Gate example
        """

        #error is the 'T' (desired output) minus 'O' (neural network output)
        error = correctOutput - predicatedOutput

        #weightCorrection is the deltaW which is learning rate * error (which is T-O)
        #When T=1 and O=0 weightCorrection will be a positive value
        #When T=0 and O=1 weightCorrection will be a negative value
        weightCorrection = self.learningRate * error

        #applying the new weights
        self.weights = self.weights + weightCorrection * selectedExample

        #applying the new bias
        self.bias = self.bias + weightCorrection

    def predict(self, testExamples):

        """
        Used to test the model once the model has been fitted. Takes an array filled with OR Gate examples as 1d arrays [[a,b], [c,d]...]
        testExamples: The testing dataset containing OR Gate examples

        Returns a 1d array with the models predictions of the output of the given OR Gate examples
        """

        results = [] #The array to be filled with predictions based on the given testing dataset

        for example in testExamples:

            #weightedSum is the dot multiplication of weights and example plus the bias
            weightedSum = np.dot(example, self.weights) + self.bias

            #Run our weighted sum through the step function to determine the prediction
            predictedOutput = self.stepFunction(weightedSum)

            #Add the prediction to the array to be returned
            results.append(predictedOutput)

        return results

# METHODS #

def createOrGateDataArray(size):

    """
    Creates an array filled with randomly generated OR Gate examples as 1d arrays
    size: The number of OR Gate examples to be created

    Returns an array filled with randomly generated OR Gate examples as 1d arrays
    """

    dataArray = []
    choice = [0, 1]

    for x in range(size):

        dataArray.append(np.random.choice(choice, 2))
    
    return dataArray

def ensureCorrectness(dataArray, dataArrayOutput):

    """
    Takes an array filled with OR Gate examples as 1d arrays and a 1d array filled with potential correct outputs for each OR Gate examples and determines if the outputs are correct or not
    dataArray       : The array filled with OR Gate examples as 1d arrays
    dataArrayOutput : The 1d array filled with potential correct outputs for each OR Gate examples
    """

    correct = 0
    incorrect = 0

    for example, answer in zip(dataArray, dataArrayOutput):
        
        if example[0] == 0 and example[1] == 0 and answer == 0:
            correct = correct + 1

        elif (example[0] == 0 and example[1] == 1 and answer == 1) or (example[0] == 1 and example[1] == 0 and answer == 1) or (example[0] == 1 and example[1] == 1 and answer == 1):
            correct = correct + 1
        
        else:
            incorrect = incorrect + 1

    if incorrect != 0:
        print("------------------------------------------")
        print(" This array has not been created properly ")
        print("------------------------------------------\n")
    
    else:
        print("------------------------------------------")
        print("   This array has been created properly   ")
        print("------------------------------------------\n")

def populateCorrectAnswers(dataArray):

    """
    Takes an array filled with 1d arrays representing OR Gate examples and determines the correct output for each
    dataArray: The array filled with 1d arrays representing OR Gate examples

    Returns a 1d array filled with the correct outputs for each OR Gate example found in the given dataArray
    """

    correctAnswers = []

    for i in dataArray:

        if i[0] == 0 and i[1] == 0:
            correctAnswers.append(0)

        else:
            correctAnswers.append(1)
    
    return correctAnswers

def testModelAccuracy(predictions, correctAnswers):

    """
    Takes an array filled with OR Gate examples and another filled with potential outputs for them. It maps the two arrays one-to-one and determines if the given outputs are correct
    predictions    : An array filled with OR Gate examples as 1d arrays. [[a,b], [c,d]...]
    correctAnswers : A 1d array filled with potential correct outputs for the predictions array
    """

    correct = 0
    incorrect = 0
    count = 1

    print("\n--------------------------------------------------")
    print("               Testing Model")
    print("--------------------------------------------------\n")

    for prediction, answer in zip(predictions, correctAnswers):

        if prediction == answer:
            print("{}. Prediction: {} | Answer: {} | Correct".format(str(count).zfill(3),prediction, answer))
            correct = correct + 1
            count = count + 1

        else:
            print("Prediction: {} | Answer: {} | Incorrect".format(str(count).zfill(3), prediction, answer))
            incorrect = incorrect + 1
            count = count + 1

    totalPredictions = correct + incorrect
    accuracy = (correct / totalPredictions) * 100

    print("\n\nThe model predicted {} correct answers out of a total of {} predictions.\nThe model achieved {}% accuracy".format(correct, totalPredictions, accuracy))

# DRIVER CODE #
    
def main():
    
    #Creating and populating the training and testing datasets
    #Split between training dataset and testing dataset will be 70% train to 30% test split
    arrayTrainExamples = createOrGateDataArray(140)
    arrayTrainExamples = np.array(arrayTrainExamples)

    arrayTestExamples = createOrGateDataArray(60)
    arrayTestExamples = np.array(arrayTestExamples)

    #Creating and populating the Output arrays for the Training and Testing datasets with the correct data
    arrayTrainOutput = populateCorrectAnswers(arrayTrainExamples)
    arrayTrainOutput = np.array(arrayTrainOutput)

    arrayTestOutput = populateCorrectAnswers(arrayTestExamples)
    arrayTestOutput = np.array(arrayTestOutput)

    #Ensuring correctness of arrays
    print()
    print("          Checking Test Sets...")
    ensureCorrectness(arrayTestExamples, arrayTestOutput)
    print("          Checking Train Sets...")
    ensureCorrectness(arrayTrainExamples, arrayTrainOutput)

    #Initializing model
    perceptronModel = Perceptron()

    #Fitting the model to our training data
    perceptronModel.fit(arrayTrainExamples, arrayTrainOutput)

    #Testing our fitted model on the testing data
    testingPredictions = perceptronModel.predict(arrayTestExamples)

    #Comparing predictions made on the testing data to our test outputs
    testModelAccuracy(testingPredictions, arrayTestOutput)
    
if __name__ == "__main__":

    main()