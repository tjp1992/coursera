from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# import linear regression model
from sklearn.linear_model import LinearRegression

# instantiate the model
model = LinearRegression()

# import the data
data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_apple_stock.csv')

# print the first 5 rows of the data
data.head()


# bool check for entry point
if __name__ == '__main__':
    # using requests library to make http request to 'https://duckduckgo.com/?q=dogs&atb=v321-1&iax=images&ia=images' to get all the images of dogs
    response = requests.get('https://duckduckgo.com/?q=dogs&atb=v321-1&iax=images&ia=images')
    # print the response
    print(response)
    # use beautiful soup to parse the response
    soup = BeautifulSoup(response.text, 'html.parser')
