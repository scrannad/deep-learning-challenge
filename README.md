# Overview

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, I have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

**EIN and NAME**—Identification columns <br />
**APPLICATION_TYPE**—Alphabet Soup application type<br />
**AFFILIATION**—Affiliated sector of industry<br />
**CLASSIFICATION**—Government organization classification<br />
**USE_CASE**—Use case for funding<br />
**ORGANIZATION**—Organization type<br />
**STATUS**—Active status<br />
**INCOME_AMT**—Income classification<br />
**SPECIAL_CONSIDERATIONS**—Special considerations for application<br />
**ASK_AMT**—Funding amount requested<br />
**IS_SUCCESSFUL**—Was the money used effectively<br />

# Results
### 1. Data Preprocessing
* Non-beneficial columns **'NAME'** and **'EIN'** were dropped
* Number of unique values for each column determined
* Values were binned, creating a cutoff point and 'Other' bin for **'APPLICATION_TYPE'** and **'CLASSIFICATION'** columns
* Data was converted to numeric with the pd.get_dummies function
* Data was split into training and testing sets

Target variable for this model was: **IS_SUCCESSFUL**. <br />
Feature variable used were: <br />
**APPLICATION_TYPE**<br />
**AFFILIATION**<br />
**CLASSIFICATION**<br />
**USE_CASE**<br />
**ORGANIZATION**<br />
**STATUS**<br />
**INCOME_AMT**<br />
**SPECIAL_CONSIDERATIONS**<br />
**ASK_AMT**<br />

### 2. Compiling, Training and Evaluating the Model
The first model was built using two hidden layers with 80 and 30 neuron splits. Feature was 42, so 80 was set as close to double the feature. Activation function on the layers was 'relu' with 'sigmoid' function on the output layer since the output of this model was binary. This resulted in a 72.9% accuracy score, which was lower than my target of 75%. 

I then attempted to optimize the model by adding an additional hidden layer, again using the 'relu' activation function. In two more models, I experimented with using the 'tanh' activation function and changing the neuron values. These models continued to produce accuracy scores below my target of 75%. 

In my final optimization, I used 'kerastuner' to automate optimization. I ran this several times, adjusting the number of maximum epochs and allowed activation functions, and allowing the kerastuner to run for up to ninety minutes per session. However, I never achieved over 74%. In my final attempt, I stopped the tuner after 56 minutes, achieving 73.9% accuracy with the best model. I then analyzed the top three models' hyperparameters as well as the accuracy and loss for each model. 

# Summary
I was able to achieve 73.8% accuracy in my optimized model using keras tuner. This was a one percent increase over my original model. I may continue to experiment with adding and removing different features try to achieve increased predictive accuracy while maintaining the shape and integrity of the dataset. 
