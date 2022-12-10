from tensorflow.keras.layers import LSTM
from keras.layers.core import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.corpus import stopwords
import re
import nltk
import streamlit as st
import matplotlib
import pandas as pd
import numpy as np
import yfinance as yf
from numpy import concatenate
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
nltk.download('stopwords')
nltk.download('vader_lexicon')
ps = PorterStemmer()
