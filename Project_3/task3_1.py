import numpy as np
import math
import matplotlib.pyplot as plt

# define the function to compute the similarity matrix values
def func_phi(s, sigma):  
    return ( np.exp(- (np.power(s, 2)) / (2 * sigma ** 2)) )       #Calculate the Gaussian function

n = 20
x = np.arange(n) + np.random.randn(n) * 0.2
y = np.random.rand(n) * 2

# define the final function to be plotted
def final_func(new_x):
    final_terms = []
    for i in range(len(weight_matrix)):     # loop over all weights
        s = new_x - x[i]
        final_terms.append(weight_matrix[i]*func_phi(s, sigma))     # calculate all terms of final function
    final_value = 0
    for i in final_terms:
        final_value += i        # Sum up all terms to obtain the final value
    return final_value

for s in [0.5,1.0,2.0,4.0]:     # different sigmas

# Initialize the similarity matrix
    similar_row = []
    similar_matrix = []
    
    sigma = s
    
    # Loop through the x values to compute the similarity matrix
    for i in x:
        for j in x:
            s = i - j                                                 #calcualte the difference between the x values
            func_value = func_phi(s,sigma)                            #call the function to calculate the gaussian value
            similar_row.append(func_value)                            #Update the row of the similarity matrix
        similar_matrix.append(similar_row)                            #update the similarity matrix
        similar_row = []                  
    
    #Invert the similarity matrix to find the weight martix
    try:    
        similar_inverse = np.linalg.inv(similar_matrix)               #compute the inverse of the similarity matrix
    except np.linalg.LinAlgError:
        pass                                                          
    else:
        weight_matrix = np.dot(similar_inverse, y)                    #compute the weight matrix based on the inverse of the similarity matrix
    
    
    xs = np.linspace(0,n,200)
    F = final_func(xs)
    plt.plot(xs,F, 'b')     # plot the calculated function in blue
    plt.plot(x, y, 'r')     # plot a linear interpolation in red for comparison
    plt.scatter(x,y,c='g')  # plot all generated points in green
    plt.savefig('result.png')
    plt.show()

