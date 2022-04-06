"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data dependencies
import pandas as pd
from utils import preprocess, predict, get_random_sample

# Load your raw data
raw = pd.read_csv("resources/test.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	#  get random tweet sample
	text = get_random_sample(raw)
	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)
	
	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
	
	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		if st.button("Get random tweet"):
			text = get_random_sample(raw)
			print()
		st.text_area("Random tweet sample", text)

		tweet_text = st.text_area("Text to predict","Type Here")
		option = st.selectbox(
		'Select a model to use',
		['Support Vector machine', 'Logistic Regression', 'Random Forest', 'Naive bayes'])
		
		if st.button("Classify"):
			# convert tweet to dataframe
			df = pd.DataFrame({'message': [tweet_text]})

			# preprocess
			processed_df = preprocess(df)

			# get predictions
			predictions = predict(processed_df)

			# select option to choose 
			output_text = {
			'0': 'Neutral', 
			'-1': 'Anti climate change', 
			'1': 'Pro Climate change', 
			'2': 'News'
			}

			st.success("Model: {} - Category: {}".format(option, output_text[str(predictions[option])]))
		
	
		
					

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
