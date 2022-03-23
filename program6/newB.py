import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("./dataset.csv")

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)