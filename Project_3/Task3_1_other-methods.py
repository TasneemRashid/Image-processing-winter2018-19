import numpy as np
import math
from scipy.interpolate import splrep, splev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def func_phi(s, sigma, model):  

    if model == 'Multi-Quadric':
        return ( np.power(np.power(s, 2) + sigma ** 2, 0.5) )
    if model == 'Inverse-Multi-Quadric':
        return ( np.power(np.power(s, 2) + sigma ** 2, -0.5) )
    if model == 'Gaussian':
        return ( np.exp(- (np.power(s, 2)) / (2 * sigma ** 2)) )
    if model == 'Thin Plate Spline Function':
        return ( np.power(s, 2) * np.log1p(s) )

n = 20
x = np.arange(n) + np.random.randn(n) * 0.2
y = np.random.rand(n) * 2
                  
def final_func(new_x,model):
    final_terms = []
    for i in range(len(weight_matrix)):
        s = np.absolute(new_x - x[i])
        final_terms.append(weight_matrix[i]*func_phi(s, sigma,model))
    final_value = 0
    for i in final_terms:
        final_value += i
    return final_value

for s in [0.5,1.0,2.0,4.0]:
    sigma = s
    print("##########################################################")
    print("sigma:", sigma)
    subplot = 1
    models = ['Gaussian','Multi-Quadric','Inverse-Multi-Quadric','Thin Plate Spline Function']
    for model in models:
        
        similar_matrix = np.zeros((n,n))
        
        similar_row = []
        similar_matrix = []
        for i in x:
            for j in x:
                s = np.absolute(i - j)
                func_value = func_phi(s,sigma,model)
                similar_row.append(func_value)
            similar_matrix.append(similar_row)
            similar_row = []
        
        try:
            similar_inverse = np.linalg.inv(similar_matrix)
        except np.linalg.LinAlgError:
            pass
        else:
            weight_matrix = np.dot(similar_inverse, y)
            
        xs = np.linspace(0,n,200)
    
        F = final_func(xs,model)
        plt.subplot(2, 2, subplot)
        plt.title(model)
        plt.plot(xs,F, 'b')
        plt.plot(x, y, 'r')  
        plt.scatter(x,y,c='g')
        plt.tight_layout()
        plt.ticklabel_format(style='plain')
        subplot += 1
    name = model + '-sigma' + str(sigma) + '.jpg'
    plt.savefig(name,dpi=1000)
    plt.show()