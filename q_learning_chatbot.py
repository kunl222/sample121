import streamlit as st
import numpy as np
import pandas as pd
from textblob import TextBlob  # Make sure to install this library using: pip install textblob
from xgb_mental_health import MentalHealthClassifier


# Initialize MentalHealthClassifier
data_path = "/Users/jaelinlee/Documents/projects/fomo/input/data.csv"
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
mental_classifier = MentalHealthClassifier(data_path)


# toxic_data_path = "/Users/jaelinlee/Documents/projects/fomo/input/toxic_text.csv"
# model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
# toxic_classifier = MentalHealthClassifier(toxic_data_path)


class QLearningChatbot:
    def __init__(self, states, actions, learning_rate=0.1, discount_factor=0.9):
        self.states = states
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_values = np.random.rand(len(states), len(actions))
        # self.q_values.at['positive', 'suggest_medical_help'] = 0
        # self.mood_history = mood_history 

    def get_action(self, current_state):
        current_state_index = self.states.index(current_state)
        # print(np.argmax(self.q_values[current_state_index, :]))
        return self.actions[np.argmax(self.q_values[current_state_index, :])]

    def update_q_values(self, current_state, action, reward, next_state):
        current_state_index = self.states.index(current_state)
        action_index = self.actions.index(action)
        next_state_index = self.states.index(next_state)

        current_q_value = self.q_values[current_state_index, action_index]
        max_next_q_value = np.max(self.q_values[next_state_index, :])

        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * max_next_q_value - current_q_value)
        self.q_values[current_state_index, action_index] = new_q_value

    def update_mood_history(self, new_mood):
        st.session_state.entered_mood.append(new_mood)
        # self.mood_history.append(new_mood)
        print(st.session_state.entered_mood)

    def check_mood_trend(self):
        print(self.mood_history)
        if len(self.mood_history) >= 2:
            recent_moods = self.mood_history[-2:]
            if recent_moods[-1] == 'positive' & recent_moods[-2] != 'positive':
                return 'increase'
            elif recent_moods[-1] == 'negative' & recent_moods[-2] != 'negative':
                return 'decrease'
            else:
                return 'unchanged'
        else:
            return 'unchanged'

###----------------------------------END OF CLASS --------------------------------------###

def detect_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    # print(f"polarity: {polarity}")
    if polarity > 0:
        return 'positive', '😊', polarity
    elif polarity == 0:
        return 'neutral', '😐', polarity
    else:
        return 'negative', '😔', polarity

# Function to display Q-table
def display_q_table(q_values, states, actions):
    q_table_dict = {'State': states}
    for i, action in enumerate(actions):
        q_table_dict[action] = q_values[:, i]
    
    q_table_df = pd.DataFrame(q_table_dict)
    return q_table_df
    

###----------------------------------START OF CHATBOT INITIALIZATION --------------------------------------###

# Define states and actions
states = ['positive', 'neutral', 'negative', 'adhd', 'anxiety', 'depression', 'not_defined']

actions = ['encouragement', 'empathy', 'suggest_medical_help']
# mood_history = []

# Initialize memory
if 'entered_text' not in st.session_state:
    st.session_state.entered_text = []
if 'entered_mood' not in st.session_state:
    st.session_state.entered_mood = []

# Create Q-learning chatbot
chatbot = QLearningChatbot(states, actions)
###----------------------------------END OF CHATBOT INITIALIZATION --------------------------------------###

###----------------------------------START OF STREAMLIT UI --------------------------------------###
# Streamlit UI
st.title("Sentiment-based Q-Learning Chatbot")



# Collect user input
user_message = st.text_input("Type your message here:")

# if submit_button:
# Append the entered value to the session variable
if user_message:
    st.session_state.entered_text.append(user_message)
    
    # Predict mental health condition
    mental_classifier.initialize_tokenizer(model_name)
    X, y = mental_classifier.preprocess_data()
    y_test, y_pred = mental_classifier.train_model(X, y)
    # input_text = "I feel anxiety whenever i am doing nothing."
    predicted_mental_category = mental_classifier.predict_category(user_message)
    print("Predicted mental health condition:", predicted_mental_category)
    
    
    # # Predict mental health category
    # predicted_category = mental_classifier.predict_mental_health_category(user_message)

    # # Display predicted category
    st.subheader("🛑 " + f"{predicted_mental_category.capitalize()}")

    # Detect sentiment of user's message
    user_sentiment, emoji, polarity = detect_sentiment(user_message)

    # Update mood history
    chatbot.update_mood_history(user_sentiment)
    # st.session_state.entered_mood.append(user_sentiment)


    sentiment_dict = {'negative': -1, 'neutral': 0, 'positive': 1}
    sentiment_dict_reverse = {-1: 'negative', 0: 'neutral', 1: 'positive'}
    user_sentiment_int = sentiment_dict.get(user_sentiment, 0)


    # Define rewards
    if polarity > 1:
        reward = 1
    elif polarity == 0:
        reward = -1
    else:
        reward = -1


    # Update Q-values and mood history
    chatbot.update_q_values(user_sentiment, chatbot.actions[0], reward, user_sentiment)

    # Get recommended action based on the updated Q-values
    recommended_action = chatbot.get_action(user_sentiment)

    # Display results
    # st.write("User mood -> Recommended AI Tone")
    st.subheader(f"{emoji} {user_sentiment.capitalize()}")
    st.subheader("➡️ " + f"{recommended_action.capitalize()}")
# st.write(f"Mood trend: {mood_trend}")

# Display Q-table
# st.subheader(f"Recommended AI Tone: {recommended_action}")
st.dataframe(display_q_table(chatbot.q_values, states, actions))

# Display mood history
st.subheader("Mood History (Recent 5):")
for mood_now in reversed(st.session_state.entered_mood[-5:]): #st.session_state.entered_text[-5:]
    st.write(f"{mood_now}")
