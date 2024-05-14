# • This program presents a small deterministic multi-layer perceptron (MLP)
#   artificial neural network, consisting of binary state neurons, analogue
#   state weights and a variable topology (the latter partly based on user input).
# • The code also implements a supervised training algorithm, back-propagation
#   or similar, reading training data from a file which contains both the input
#   binary vectors as well as the desired classification labels.
# • All these are executed through these menu options:
#    1. Read the labelled text data file, display the first 5 lines
#    2. Choose the size of the hidden layers of the MLP topology (e.g. 6-?-?-2)
#    3. Choose the size of the training step (0.001 - 0.5, [ENTER] for adaptable)
#    4. Train on 80% of labeled data, display progress graph
#    5. Classify the unlabeled data, output training report and confusion matrix
#    6. Exit
# Copyright: GNU Public License http://www.gnu.org/licenses/
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# a class for the neural network
class NeuralNetwork:

    def __init__(self):
        self.layers = []
        self.weights = []
        self.biases = []

    # a function that adds the neurons of each hidden layer
    def add_layer(self, num_neurons):
        self.layers.append(num_neurons)

        if len(self.layers) >= 2:
            prev_layer_neurons = self.layers[-2]
            current_layer_neurons = self.layers[-1]
            self.weights.append(np.random.random((current_layer_neurons, prev_layer_neurons)) - 0.50)
            self.biases.append(np.zeros((current_layer_neurons, 1)))

    # a function that calculates the forward propagation
    def forward_propagation(self, X):
        activations = [X.T]
        zs = []

        for i in range(len(self.layers) - 1):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            a = sigmoid(z)
            zs.append(z)
            activations.append(a)

        return activations, zs

    # a function that calculates the backward propagation
    def backward_propagation(self, X, y, activations, zs):
        m = X.shape[0]
        grads = []

        # Perform backward propagation
        for i in range(len(self.layers) - 2, -1, -1):
            if i == len(self.layers) - 2:
                delta = (activations[-1] - y.T) * sigmoid_derivative(zs[i])
            else:
                delta = np.dot(self.weights[i + 1].T, delta) * sigmoid_derivative(zs[i])

            dW = (1 / m) * np.dot(delta, activations[i].T)
            db = (1 / m) * np.sum(delta, axis=1, keepdims=True)
            grads.insert(0, (dW, db))  # prepend the gradients instead of appending

        return grads

    # a function for training the dataset
    def train(self, X, y, learning_rate, epochs):
        losses = []
        weight_changes = []

        for epoch in range(epochs):
            activations, zs = self.forward_propagation(X)
            grads = self.backward_propagation(X, y, activations, zs)

            # update weights and biases
            for i in range(len(self.weights)):
                self.weights[i] -= learning_rate * grads[i][0]
                self.biases[i] -= learning_rate * grads[i][1]

            loss = np.mean(-y.T * np.log(activations[-1]) - (1 - y.T) * np.log(1 - activations[-1]))
            losses.append(loss)

            if epoch > 0:
                prev_weights = np.concatenate([w.flatten() for w in weight_changes[-1]])
                current_weights = np.concatenate([w.flatten() for w in self.weights])
                weight_change = np.mean(np.abs(current_weights - prev_weights))
                weight_changes.append([np.copy(w) for w in self.weights])
                print(f"Epoch {epoch}: Average Weight Change: {weight_change}")
            else:
                weight_changes.append([np.copy(w) for w in self.weights])

        return losses, weight_changes

    # a function that produces the graph for weight changes and epochs
    def visualize_weights(self, weight_changes):
        num_layers = len(weight_changes[0])
        num_epochs = len(weight_changes)

        # Plot all weights together in one diagram
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("All Weights")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weight")

        for layer_idx in range(num_layers):
            for neuron in range(self.weights[layer_idx].shape[1]):
                weights = [weight_changes[epoch][layer_idx][:, neuron] for epoch in range(num_epochs)]
                ax.plot(range(num_epochs), weights, label=f"Layer {layer_idx + 1}, Neuron {neuron + 1}")

        ax.legend()
        plt.show()

    # a function to classify the unlabeled data
    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        return activations[-1]
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# a function that displays the information of the menu options
def menu():
    # prints the information for the menu options
    while True:
        print("====================================================================================", "\n",
              "USER MENU:  MLP CLASSIFICATION OF THE BANK NOTE IDENTIFICATION DATA SET (UCI REPOSITORY)", "\n",
              "====================================================================================", "\n",
              "1. Read the labelled text data file, display the first 5 lines", "\n",
              "2. Choose the size of the hidden layers of the MLP topology (e.g. 6-?-?-2)", "\n",
              "3. Choose the size of the training step (0.001 - 0.5, [ENTER] for adaptable)", "\n",
              "4. Train on 80% of labeled data, display progress graph", "\n",
              "5. Classify the unlabeled data, output training report and confusion matrix", "\n",
              "6. Exit the program")
        try:
            # accepts the input from the user
            option = int(input("Please select an option: "))
            break

        # in case the user enters an invalid option
        except ValueError:
            print("Invalid input. Please select a valid menu option: ")

        # if any other error occurs
        except Exception as e:
            print("An error occurred: ", str(e))

    return option
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# a function to print the results of menu option 1
def menu_option_1():
    print("---------------------------------------------------------------------------------")
    print("You selected menu option 1.")
    print("   This menu option reads the data from the file and assigns column")
    print("   names to them. Then it displays the first 5 rows of this dataset.")
    print()

    # the location of the dataset
    data = "original_labeled_data.txt"

    # assigns column names to the dataset
    labels = ['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class']

    # reads the dataset to a pandas dataframe
    bank_data = pd.read_csv(data, names=labels)

    # displays the first 5 rows of the dataset along with their column names
    print("Printing the first few lines of the data set:\n", bank_data.head())
    print()

    # writes the data of the original_labeled_data.txt to a new file with labels
    bank_data.to_csv("training_data.txt", index=False)

    # informs the user where to find the results of this menu option
    print("You can also find the entire dataset with column names in training_data.txt file.")
    print("---------------------------------------------------------------------------------")

    return bank_data
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# a function to print the results of menu option 2
def menu_option_2():
    print("---------------------------------------------------------------------------------")
    print("You selected menu option 2.")
    print("   This menu option asks for the amount of hidden layers that the ANN should have.")
    print("   It also asks for the amount of neurons that each hidden layer should have.")
    print()

    # declare and initialise variables
    neurons_in_layer = ()
    size_topography = 0

    while True:
        try:
            # accepts the input from the user
            size_topography = int(input("Please select how many hidden layers the program should have: "))

            # if the size of the topography is <2 or >8
            if size_topography < 2 or size_topography > 8:
                print("Please enter between 2 to 8 hidden layers, no more and no less.")
            else:
                break

        # in case the user enters an invalid option
        except ValueError:
            print("Invalid input. Please select a valid size for MLP topography.")

        # if any other error occurs
        except Exception as e:
            print("An error occurred: ", str(e))

    # a loop for the neurons of each hidden layer
    for i in range(size_topography):
        while True:
            try:
                # accepts the input for the user
                print("How many neurons should layer ", i, " have?")
                num_neurons = int(input("Please select the amount of neurons: "))

                # if the amount of neurons is >=4 and <=10
                if 3 <= num_neurons <= 10:
                    neurons_in_layer += (num_neurons,)
                    break

                # if the amount of neurons is not within those limits
                else:
                    print("Please enter between 4 to 10 neurons for the hidden layers.")

            # in case the user enters an invalid input
            except ValueError:
                print("Invalid input. Please enter a valid number of neurons.")

    print("---------------------------------------------------------------------------------")

    return neurons_in_layer
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# a function to print the results of menu option 3
def menu_option_3():
    print("---------------------------------------------------------------------------------")
    print("You selected menu option 3.")
    print("   This menu option asks for the size of the training step that the ANN should have.")
    print()
    while True:
        option_for_training_step = str("Enter one of the following options:" + "\n" +
                                       "Enter '1' for the size of the training size to be 0.001." + "\n" +
                                       "Enter '2' for the size of the training step to be 0.5." + "\n" +
                                       "Enter '3' to choose a size between the range 0.001-0.5." + "\n" +
                                       "Press 'enter' to have an adaptable size of the training step." + "\n")

        # accepts the user input
        option = input(option_for_training_step)

        # as long as the user enter a valid input
        if option == "1" or option == "2" or option == "3" or option == "":
            break

    # if the user enters 1
    if option == "1":
        return 0.001

    # if the user enters 2
    elif option == "2":
        return 0.5

    # if the user enters 'enter'
    elif option == "":
        return "adaptive"

    # if the user enters 3
    else:
        while True:
            try:
                training_step = float(input("Enter a number between 0.001 and 0.5: "))
                if 0.001 <= training_step <= 0.5:
                    break

            # in case the user enters an invalid input
            except ValueError:
                print("Invalid input. Please enter a positive number between those values or hit enter.")

            # if any other error occurs
            except Exception as e:
                print("An error occurred: ", str(e))

        print("---------------------------------------------------------------------------------")

        return training_step
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# a function to print the results of menu option 4
def menu_option_4(bank_data, neurons_in_layer, training_step):
    print("---------------------------------------------------------------------------------")
    print("You selected menu option 4.")
    print("   This menu option trains on 80% of the labeled data and displays the progress graph.")
    print()

    # assigns column names to the dataset
    labels = ['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class']

    # assign data from first four columns to X variable
    X = bank_data[labels[:-1]].values

    # assign data from first fifth columns to y variable
    y = bank_data[labels[-1]].values

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # scale the features (optional)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # calls the NeuralNetwork function and assigns it to ann

    # adds 4 neurons to the first layer, which is the input layer
    myAnn.add_layer(4)

    # a loop for the amount of neurons in each hidden layer
    for layer in neurons_in_layer:
        myAnn.add_layer(layer)

    # adds 1 neurons to the last layer, which is the output layer
    myAnn.add_layer(1)

    # if the user pressed enter and the training step is adaptive
    if training_step == "adaptive":
        learning_rate = 0.001 + (0.5 - 0.001) * np.random.random()
        losses, weight_changes = myAnn.train(X_train_scaled, y_train, epochs=20000,
                                           learning_rate=learning_rate)

    # if the training step is a number the user entered
    else:
        losses, weight_changes = myAnn.train(X_train_scaled, y_train, epochs=20000,
                                           learning_rate=training_step)

    # calls the function that makes the graph for the weight change and the epochs
    myAnn.visualize_weights(weight_changes)
    plt.plot(range(20000), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Total losses")
    plt.title("Total losses per Epoch")
    plt.show()
    # takes all the columns of the dataset apart from the last one
    testing_data_unlabeled = bank_data.iloc[:, :-1]

    # saves the testing data without labels to testing_data_unlabeled.txt file
    testing_data_unlabeled.to_csv("testing_data_unlabeled.txt", index=False, header=False)

    print()
    # informs the user where to find the unlabeled testing data
    print("You can also find the unlabeled testing data in the testing_data_unlabeled.txt file.")
    print()

    # takes all the columns of the dataset apart from the last one
    testing_data_labeled = bank_data.iloc[:, :-1]

    # saves the testing data with labels to testing_data_unlabeled.txt file
    testing_data_labeled.to_csv("testing_data_labeled.txt", index=False)

    # informs the user where to find the labeled testing data
    print("You can also find the labeled testing data in the testing_data_labeled.txt file.")

    print("---------------------------------------------------------------------------------")

    return X_test_scaled, y_test
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# a function to print the results of menu option 5
def menu_option_5(X_test_scaled, y_test):
    print("---------------------------------------------------------------------------------")
    print("You selected menu option 5.")
    print("   This menu option classifies the unlabeled data, outputs the training report and confusion matrix.")
    print()

    # load the training data
    training_data = pd.read_csv("training_data.txt")

    # load the testing data
    testing_data_unlabeled = pd.read_csv("testing_data_unlabeled.txt", header=None)

    # load the testing data labels
    testing_data_labeled = pd.read_csv("testing_data_labeled.txt")

    # finds the accuracy
    predictions = myAnn.predict(X_test_scaled)
    predictions = (predictions > 0.5).astype(int).flatten()
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy}")
    my_matrix = confusion_matrix(y_test, predictions)
    print(my_matrix)
    print(classification_report(y_test, predictions))

    # writes the predictions to the output_data.txt file
    with open("output_data.txt", "w") as output_file:
        output_file.write("Class")
        output_file.write("\n")
        for prediction in predictions:
            output_file.write(str(prediction) + "\n")

    plt.imshow(my_matrix, cmap='Blues')
    for i in range(len(my_matrix)):
        for j in range(len(my_matrix[i])):
            plt.text(j, i, str(my_matrix[i][j]), ha='center', va='center', color='black')

    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ['Class_0', 'Class_1'])
    plt.yticks([0, 1], ['Class_0', 'Class_1'])
    plt.title("Confusion Matrix")
    plt.show()

    print("---------------------------------------------------------------------------------")
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# declare and initialise variables
bank_data = []

global myAnn

myAnn = NeuralNetwork()

option = menu()
# declare and initialise variables as false for when the menu options are executed
menu_option_1_executed = False
menu_option_2_executed = False
menu_option_3_executed = False
menu_option_4_executed = False

# all the menu options, apart from menu option 1 and 6, will only execute if the
# previous menu options have been executed in order
while option != 6:

    # if the user chooses menu option 1
    if option == 1:
        # calls the function for menu option 1
        bank_data = menu_option_1()
        print()
        # change the value of menu_option_1_executed to True
        menu_option_1_executed = True

    # if the user chooses menu option 2
    elif option == 2:
        # it will execute menu option 2 only if menu option 1 has already been executed
        if menu_option_1_executed:
            # calls the function for menu option 2
            neurons_in_layer = menu_option_2()
            print()
            # change the value of menu_option_2_executed to True
            menu_option_2_executed = True
        # in case that menu option 1 has not been executed
        else:
            # informs the user to execute menu option 1
            print("Please execute menu option 1 first and then proceed to this menu option.")
            print()

    # if the user chooses menu option 3
    elif option == 3:
        # it will execute menu option 3 only if the previous menu options have already been executed
        if menu_option_1_executed and menu_option_2_executed:
            # calls the function for menu option 3
            training_step = menu_option_3()
            print()
            # change the value of menu_option_3_executed to True
            menu_option_3_executed = True
        # in case that the previous menu options have not been executed
        else:
            # informs the user to execute the previous menu option
            print("Please execute the other menu options first and then proceed to this menu option.")
            print()

    # if the user chooses menu option 4
    elif option == 4:
        # it will execute menu option 4 only if the previous menu option have already been executed
        if menu_option_1_executed and menu_option_2_executed and menu_option_3_executed:
            # calls the function for menu option 4
            X_test_scaled, y_test = menu_option_4(bank_data, neurons_in_layer, training_step)
            print()
            # change the value of menu_option_4_executed to True
            menu_option_4_executed = True
        # in case that the previous menu options have not been executed
        else:
            # informs the user to execute the previous menu options
            print("Please execute the previous menu options first and then proceed to this menu option.")
            print()

    # if the user chooses menu option 5
    elif option == 5:
        # it will execute menu option 5 only if menu option 1 and 3 have already been executed
        if menu_option_1_executed and menu_option_2_executed and menu_option_3_executed and menu_option_4_executed:
            # calls the function for menu option 5
            menu_option_5(X_test_scaled, y_test)
            print()
        # in case that the previous menu options have not been executed
        else:
            # informs the user to execute the previous menu options
            print("Please execute the previous menu options first and then proceed to this menu option.")
            print()

    # if the user chooses menu option 6
    elif option == 6:
        print("You selected option 6.")
        print("Exiting the program...", "\n")
    # if the user enter an invalid input for the menu option
    else:
        print("Invalid entry. Please enter \"1\" , \"2\", \"3\", \"4\", \"5\" or \"6\"", "\n")

    option = menu()

print()
print("---------------------------------------------------------------------------------")
print("You selected option 6!")
print("Exiting...")
print("---------------------------------------------------------------------------------")
print()

# terminates the program
os.system("pause")
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
