import streamlit as st
import numpy as np
import pandas as pd
import torch
import pickle
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import torch.optim as optim

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class Dataframe:
    def __init__(self):
        self.ratings = pd.read_csv('ratings.csv')
        self.ratings = self.ratings[['userId', 'movieId', 'rating']]
        self.ratings_df = self.ratings.groupby(['userId', 'movieId']).agg('max')
        self.movie_list = pd.read_csv('movies.csv')
        self.tags = pd.read_csv('tags.csv')
        self.genres = self.movie_list['genres']
        self.genre_list = ""
        for index, row in self.movie_list.iterrows():
            self.genre_list += row.genres + "|"
        self.genre_list_split = self.genre_list.split('|')
        self.new_list = list(set(self.genre_list_split))
        self.new_list.remove('')
        self.movies_with_genres = self.movie_list.copy()
        for genre in self.new_list:
            self.movies_with_genres[genre] = self.movies_with_genres.apply(lambda _: int(genre in _.genres), axis=1)
        self.no_of_users = len(self.ratings['userId'].unique())
        self.no_of_movies = len(self.ratings['movieId'].unique())
        self.sparsity = round(1.0 - len(self.ratings) / (1.0 * (self.no_of_movies * self.no_of_users)), 3)
        self.avg_movie_rating = pd.DataFrame(self.ratings.groupby('movieId')['rating'].agg(['mean', 'count']))
        self.avg_rating_all = self.ratings['rating'].mean()
        self.min_reviews = 30
        self.movie_score = self.avg_movie_rating.loc[self.avg_movie_rating['count'] > self.min_reviews]
        self.ratings_movies = pd.merge(self.ratings, self.movie_list, on='movieId')
        self.flag = False

    def weighted_rating(self, x, m=None, C=None):
        if m is None:
            m = self.min_reviews
        if C is None:
            C = self.avg_rating_all
        v = x['count']
        R = x['mean']
        return (v / (v + m) * R) + (m / (m + v) * C)

    def best_movies_by_genre(self, genre, top_n):
        movie_score_copy = self.movie_score.copy()
        if not self.flag:
            movie_score_copy['weighted_score'] = movie_score_copy.apply(lambda x: self.weighted_rating(x), axis=1)
            movie_score_copy = pd.merge(movie_score_copy, self.movies_with_genres, on='movieId')
            self.flag = True
        return pd.DataFrame(
            movie_score_copy.loc[(movie_score_copy[genre] == 1)].sort_values(['weighted_score'], ascending=False)[
                ['title', 'count', 'mean', 'weighted_score']][:top_n])

    def get_other_movies(self, movie_name, top_n):
        df_movie_users_series = self.ratings_movies.loc[self.ratings_movies['title'] == movie_name]['userId']
        df_movie_users = pd.DataFrame(df_movie_users_series, columns=['userId'])
        other_movies = pd.merge(df_movie_users, self.ratings_movies, on='userId')
        other_users_watched = pd.DataFrame(other_movies.groupby('title')['userId'].count()).sort_values('userId', ascending=False)
        if len(other_users_watched) > 0:
            other_users_watched['perc_who_watched'] = round(other_users_watched['userId'] * 100 / other_users_watched['userId'].iloc[0], 1)
        return other_users_watched[:top_n]

    def get_book_recommendations(self, year, top_n):
        books = pd.read_csv('books.csv', sep=';', encoding='latin-1')
        recommendations = books.loc[books['Year-Of-Publication'] == year, ['Book-Title', 'Book-Author']].head(top_n)
        return recommendations

df = Dataframe()

class Dataset:
    def __init__(self):
        with open("recommendintents.json") as file:
            self.filedata = json.load(file)
        self.words = []
        self.labels = []
        self.docs_x = []
        self.docs_y = []
        self.stemmer = LancasterStemmer()
        self.stop_words = set(stopwords.words('english'))
        for intent in self.filedata['intents']:
            for pattern in intent['patterns']:
                wrds = word_tokenize(pattern)
                wrds = [w for w in wrds if w not in self.stop_words]
                self.words.extend(wrds)
                self.docs_x.append(wrds)
                self.docs_y.append(intent["tag"])
            if intent['tag'] not in self.labels:
                self.labels.append(intent['tag'])
        self.words = [self.stemmer.stem(w.lower()) for w in self.words if w != "?"]
        self.words = sorted(list(set(self.words)))
        self.labels = sorted(self.labels)

    def get_training_data(self):
        training = []
        output = []
        out_empty = [0 for _ in range(len(self.labels))]
        for x, doc in enumerate(self.docs_x):
            bag = []
            wrds = [self.stemmer.stem(w.lower()) for w in doc]
            for w in self.words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)
            output_row = out_empty[:]
            output_row[self.labels.index(self.docs_y[x])] = 1
            training.append(bag)
            output.append(output_row)
        training = np.array(training)
        output = np.array(output)
        from sklearn.utils import shuffle
        training, output = shuffle(training, output, random_state=0)
        return training, output

    def bag_of_words(self, s):
        bag = [0 for _ in range(len(self.words))]
        s_words = word_tokenize(s)
        s_words = [self.stemmer.stem(word.lower()) for word in s_words]
        s_words = [w for w in s_words if w not in self.stop_words]
        for se in s_words:
            for i, w in enumerate(self.words):
                if w == se:
                    bag[i] = 1
        return np.array(bag)

try:
    data = Dataset()
    training, output = data.get_training_data()
except Exception as e:
    st.error(f"Error initializing the chatbot: {str(e)}")
    data = None
    training = None
    output = None

class Net(nn.Module):
    def __init__(self, training, output):
        super().__init__()
        self.fc1 = nn.Linear(len(training[0]), 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, len(output[0]))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

class Train_Model:
    def __init__(self, training, output):
        self.optimizer = optim.Adam(net.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()
        self.X = torch.tensor(training, dtype=torch.float32)
        self.Y = torch.tensor(output, dtype=torch.float32)

    def start_training(self, net, BATCH_SIZE, EPOCHS):
        for epoch in range(EPOCHS):
            for i in range(0, len(self.X), BATCH_SIZE):
                batch_X = self.X[i:i + BATCH_SIZE].view(-1, len(self.X[0]))
                batch_Y = self.Y[i:i + BATCH_SIZE]
                net.zero_grad()
                outputs = net(batch_X)
                loss = self.loss_function(outputs, batch_Y)
                loss.backward()
                self.optimizer.step()
            if epoch % 100 == 0:  # Only print every 100 epochs
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        return net

if training is not None and output is not None:
    net = Net(training, output)
    tr = Train_Model(training, output)
    net = tr.start_training(net, 8, 500)
else:
    net = None

user_input_counter = 0

def get_user_input(prompt, key_prefix):
    global user_input_counter
    user_input_counter += 1
    return st.text_input(prompt, key=f"{key_prefix}_{user_input_counter}")

def get_genre_from_text(text):
    # Common genres in movies.csv
    genres = [
        'Action', 'Comedy', 'Drama', 'Thriller', 'Horror', 
        'Romance', 'Sci-Fi', 'Adventure', 'Documentary', 
        'Animation', 'Fantasy', 'Crime', 'Mystery'
    ]
    
    # Convert text to lowercase for matching
    text = text.lower()
    
    # Check for genre mentions in the text
    for genre in genres:
        if genre.lower() in text:
            return genre
    return None

def get_top_rated_movies(df, n=5):
    """Get the top n rated movies based on weighted rating"""
    if not df.flag:
        movie_score_copy = df.movie_score.copy()
        movie_score_copy['weighted_score'] = movie_score_copy.apply(lambda x: df.weighted_rating(x), axis=1)
        movie_score_copy = pd.merge(movie_score_copy, df.movies_with_genres[['movieId', 'title']], on='movieId')
        df.flag = True
    
    return pd.DataFrame(
        movie_score_copy.sort_values(['weighted_score'], ascending=False)[
            ['title', 'count', 'mean', 'weighted_score']][:n]
    )

# Add this near the top of the file after imports
def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def add_to_chat_history(role, content):
    st.session_state.chat_history.append({"role": role, "content": content})

def display_chat_message(role, content):
    if role == "user":
        message_alignment = "flex-end"
        background = "linear-gradient(135deg, rgba(127, 90, 240, 0.2), rgba(127, 90, 240, 0.1))"
        border_radius = "20px 20px 0px 20px"
    else:
        message_alignment = "flex-start"
        background = "linear-gradient(135deg, rgba(44, 182, 125, 0.2), rgba(44, 182, 125, 0.1))"
        border_radius = "20px 20px 20px 0px"
    
    # Escape HTML characters in content
    content = (
        content.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
    
    st.markdown(
        f"""
        <div style="display: flex; justify-content: {message_alignment};">
            <div class="message animate-fade-in" style="
                background: {background};
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.18);
                padding: 12px 18px;
                border-radius: {border_radius};
                margin: 8px;
                max-width: 70%;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: #FFFFFE;
                white-space: pre-wrap;
                line-height: 1.4;
                font-size: 15px;
                position: relative;
                transition: all 0.3s ease;
            ">
                {content}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Update the chat function
def chat():
    initialize_session_state()
    
    if data is None or net is None:
        st.error("Chatbot is not properly initialized. Please check the system requirements.")
        return

    # Display chat header
    st.markdown(
        """
        <div style="
            background-color: #075e54;
            padding: 15px;
            border-radius: 10px;
            color: white;
            margin-bottom: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        ">
            <h3 style="margin: 0;">🤖 Movie & Book Recommender Chat</h3>
            <p style="margin: 5px 0 0 0; font-size: 0.9em;">Online</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display chat history
    for message in st.session_state.chat_history:
        display_chat_message(message["role"], message["content"])

    # Chat input with label
    user_input = st.text_input(
        "Message",  # Add a label to fix the warning
        key=f"chat_input_{user_input_counter}",
        placeholder="Type your message here...",
        label_visibility="collapsed"  # Hide the label but keep it for accessibility
    )

    if user_input.lower() == "quit":
        add_to_chat_history("user", "Goodbye! 👋")
        add_to_chat_history("assistant", "Chat ended. Have a great day! 😊")
        return

    if user_input:
        add_to_chat_history("user", user_input)

        try:
            with torch.no_grad():
                input_tensor = torch.tensor(data.bag_of_words(user_input), dtype=torch.float32).view(1, len(data.words))
                results = net(input_tensor)[0]
                results_index = np.argmax(results.detach().numpy())
                tag = data.labels[results_index]
                confidence = float(results[results_index])

                if confidence > 0.6:
                    if tag == "genre":
                        genre = get_genre_from_text(user_input)
                        if genre:
                            recommendations = df.best_movies_by_genre(genre, 5)
                            if not recommendations.empty:
                                genre_responses = None
                                for intent in data.filedata["intents"]:
                                    if intent["tag"] == "genre":
                                        genre_responses = intent["responses"]
                                        break
                                
                                if genre_responses:
                                    response = random.choice(genre_responses)
                                    add_to_chat_history("assistant", response)
                                
                                recommendations_text = "Here are your recommendations:\n\n"
                                for _, row in recommendations.iterrows():
                                    recommendations_text += f"🎬 {row['title']}\n"
                                    recommendations_text += f"   Rating: {'⭐' * int(row['mean'])}\n"
                                    recommendations_text += f"   Reviews: {row['count']}\n\n"
                                
                                add_to_chat_history("assistant", recommendations_text)
                            else:
                                add_to_chat_history("assistant", f"Sorry, I couldn't find any movies in the {genre} genre.")
                        else:
                            add_to_chat_history("assistant", "Could you please specify which genre you're interested in?")
                    
                    elif tag == "rating_based":
                        recommendations = get_top_rated_movies(df)
                        if not recommendations.empty:
                            rating_responses = None
                            for intent in data.filedata["intents"]:
                                if intent["tag"] == "rating_based":
                                    rating_responses = intent["responses"]
                                    break
                            
                            if rating_responses:
                                add_to_chat_history("assistant", random.choice(rating_responses))
                            
                            recommendations_text = "Here are the top rated movies:\n\n"
                            for _, row in recommendations.iterrows():
                                recommendations_text += f"🎬 {row['title']}\n"
                                recommendations_text += f"   Rating: {'⭐' * int(row['mean'])}\n"
                                recommendations_text += f"   Reviews: {row['count']}\n\n"
                            
                            add_to_chat_history("assistant", recommendations_text)
                        else:
                            add_to_chat_history("assistant", "Sorry, I couldn't find any top-rated movies at the moment.")
                    
                    elif tag == "contentbasedrated":
                        add_to_chat_history("assistant", "What's your favorite movie? I'll recommend similar ones!")
                        mov = get_user_input("Movie name:", "movie_name")
                        if mov:
                            recommendations = df.get_other_movies(mov, 5)
                            if not recommendations.empty:
                                recommendations_text = f"Fans of '{mov}' also watched:\n\n"
                                for idx, row in recommendations.iterrows():
                                    recommendations_text += f"🎬 {idx}\n"
                                    recommendations_text += f"   Watched by: {row['perc_who_watched']}% of fans\n\n"
                                add_to_chat_history("assistant", recommendations_text)
                            else:
                                add_to_chat_history("assistant", "Sorry, I couldn't find recommendations for this movie.")
                    
                    elif tag == "book_preference":
                        add_to_chat_history("assistant", "I can help you find books from a specific year! What year are you interested in?")
                        year = st.number_input("Year:", min_value=1900, max_value=2024, key=f"year_{user_input_counter}")
                        if year:
                            recommendations = df.get_book_recommendations(year, 5)
                            if not recommendations.empty:
                                recommendations_text = f"Here are some books from {year}:\n\n"
                                for _, row in recommendations.iterrows():
                                    recommendations_text += f"📚 {row['Book-Title']}\n"
                                    recommendations_text += f"   Author: {row['Book-Author']}\n\n"
                                add_to_chat_history("assistant", recommendations_text)
                            else:
                                add_to_chat_history("assistant", f"Sorry, I couldn't find any books from {year}.")
                    
                    else:
                        for intent in data.filedata["intents"]:
                            if intent['tag'] == tag:
                                responses = intent['responses']
                                add_to_chat_history("assistant", random.choice(responses))
                                break
                else:
                    add_to_chat_history("assistant", f"I'm not very confident about this ({confidence:.2%}). Could you please rephrase your question?")
        
        except Exception as e:
            add_to_chat_history("assistant", f"An error occurred: {str(e)}\nPlease try asking something else.")

    # Add a clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

def display_movie_page():
    st.title("🎥 Movies")
    st.write("Check out some of the latest movies!")

    # Assuming you have a list of movie image URLs
    movie_images = [
        "https://cinemusefilms.com/wp-content/uploads/2021/12/nitram-777-x-630-copy.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2021/06/death-of-a-ladies-man.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2021/05/377-the-father.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2017/02/160-hidden-figures.jpg",
        # Add more movie image URLs as needed
        "https://cinemusefilms.com/wp-content/uploads/2021/03/375-another-round.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2021/03/375-summerland-1.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2021/02/372-my-salinger-year.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2021/02/ammonite.jpg",
        # Add more movie image URLs as needed
        "https://cinemusefilms.com/wp-content/uploads/2021/02/373-news-of-the-world.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2021/01/369-promising-young-woman.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2020/12/on-the-rocks.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2021/01/nomadland.jpg",
        # Add more movie image URLs as needed
        "https://cinemusefilms.com/wp-content/uploads/2020/09/366-made-in-italy.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2020/08/365-babyteeth.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2020/11/367-corpus-christi.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2020/07/shirley.jpg",
          # Add more movie image URLs as needed
        "https://cinemusefilms.com/wp-content/uploads/2020/06/362-sorry-we-missed-you.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2020/08/the-king-of-staten-island.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2020/03/invisible-man-2020-poster.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2020/02/360-military-wives.jpg",
          # Add more movie image URLs as needed
        "https://cinemusefilms.com/wp-content/uploads/2020/03/361-downhill.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2020/02/358-bombshell.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2020/01/357-a-beautiful-day.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2020/02/359-richard-jewell.jpg",
         # Add more movie image URLs as needed
        "https://cinemusefilms.com/wp-content/uploads/2018/12/children-act.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2018/11/boy-erased.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2018/10/304-first-man.jpg",
        "https://cinemusefilms.com/wp-content/uploads/2018/10/302-custody.jpg",
    ]

    num_columns = 4  # Number of columns to display the movie images
    image_size = 200  # Size of each movie image

    # Movie names to be added under each picture
    movie_names = [
        "Nitram",
        "Death of a Ladies' Man",
        "The Father",
        "Hidden Figures",
        # Add more movie names as needed
        "Another Round",
        "Summerland",
        "My Salinger Year",
        "Ammonite",
        # Add more movie names as needed
        "News of the World",
        "Promising Young Woman",
        "On the Rocks",
        "Nomadland",
        # Add more movie names as needed
        "Made in Italy",
        "Babyteeth",
        "Corpus Christi",
        "Shirley",
          # Add more movie names as needed
        "Sorry We Missed You",
        "The King of Staten Island",
        "The Invisible Man",
        "Military Wives",
          # Add more movie names as needed
        "Downhill",
        "Bombshell",
        "A Beautiful Day in the Neighborhood",
        "Richard Jewell",
         # Add more movie names as needed
        "Children Act",
        "Boy Erased",
        "First Man",
        "Custody",
    ]

    # Calculate the number of rows based on the number of movie images and columns
    num_rows = len(movie_images) // num_columns + (len(movie_images) % num_columns > 0)

    # Display movie images in a grid layout
    for i in range(num_rows):
        cols = st.columns(num_columns)  # Create columns for each row
        for j in range(num_columns):
            image_index = i * num_columns + j
            if image_index < len(movie_images):
                cols[j].image(movie_images[image_index], use_container_width=True)
                cols[j].write(movie_names[image_index])
            else:
                cols[j].empty()  # If there are less than 4 movies in the last row, fill empty columns
    
def display_book_page():
    st.title("📗 Books")
    st.write("Check out some of the latest books!")


    # Assuming you have a list of book cover image URLs
    book_images = [
        "https://m.media-amazon.com/images/I/413z8uYYpdL._SY445_SX342_.jpg",
        "https://m.media-amazon.com/images/I/41DK3BVS1OL._SY445_SX342_.jpg",
        "https://m.media-amazon.com/images/I/41I9gyJBlQL._SY445_SX342_.jpg",
        "https://m.media-amazon.com/images/I/41ALFE8zbpL._SY445_SX342_.jpg",
        "https://m.media-amazon.com/images/I/41r+RIupVbL._SY445_SX342_.jpg",
        "https://m.media-amazon.com/images/I/31uj-12L7VL._SY445_SX342_.jpg",

        "https://m.media-amazon.com/images/I/31K216eELVL._SY445_SX342_.jpg",
        "https://m.media-amazon.com/images/I/311BxzpRSbL._SY445_SX342_.jpg",
        "https://m.media-amazon.com/images/I/31S-l152KTL._SY445_SX342_.jpg",
        "https://m.media-amazon.com/images/I/31g3utJiX+L._SY445_SX342_.jpg",
        "https://m.media-amazon.com/images/I/31y+d64CbFL._SY445_SX342_.jpg",
        "https://m.media-amazon.com/images/I/41bByOS56uL._SY445_SX342_.jpg",


        "https://m.media-amazon.com/images/I/31P7WZ9sciL._SY445_SX342_.jpg",
        "https://m.media-amazon.com/images/I/41Jl-jm5xjL._SY445_SX342_.jpg",
        "https://m.media-amazon.com/images/I/41+-FFnlusL._SY445_SX342_.jpg",

        "https://m.media-amazon.com/images/I/41AZ8zr8fWL._SY445_SX342_.jpg",
        "https://m.media-amazon.com/images/I/41lXpiuZqeL._SY445_SX342_.jpg",
        "https://m.media-amazon.com/images/I/41LuUlddrhL._SY445_SX342_.jpg",

        "https://m.media-amazon.com/images/I/81Dd8VnZkhL._SY425_.jpg",
        "https://m.media-amazon.com/images/I/41BmfVVsEvL._SY445_SX342_.jpg",
        "https://m.media-amazon.com/images/I/5177eLEs+YL._SY445_SX342_.jpg",
        "https://m.media-amazon.com/images/I/41MSk1PGEdL._SY445_SX342_.jpg",
        "https://m.media-amazon.com/images/I/31iH5BDxmCL._SY445_SX342_.jpg",
        "https://m.media-amazon.com/images/I/418V2HJSGWL._SY445_SX342_.jpg",
    ]

    num_columns = 4  # Number of columns to display the book images
    image_size = 300  # Size of each book image
    # Book names to be added under each picture
    book_names = [
        "How to Talk to Anybody",
        "Atomic Habits",
        "The Teacher",
        "surrounded by Idiots",
        # Add more movie names as needed
        "Make Me Lose",
        "EMMA",
        "Persuation",
        "Hermès",
        # Add more movie names as needed
        "Channel",
        "Dior",
        "Louis Vuitton",
        "Master your Emotions",
        # Add more movie names as needed
        "The Book of Tea",
        "Stoicism",
        "Awakening",
        "Master Your Destiny",
        # Add more movie names as needed
        "30 days",
        "The Book Thief",
        "Odd Numbers",
        "A little life",
        # Add more movie names as needed
        "The Silent Patient",
        "The Bell Jar",
        "YellowFace",
        "Things We Left Behind",
    ]

    # Calculate the number of rows based on the number of book images and columns
    num_rows = len(book_images) // num_columns + (len(book_images) % num_columns > 0)

    # Display book images in a grid layout
    for i in range(num_rows):
        cols = st.columns(num_columns)  # Create columns for each row
        for j in range(num_columns):
            image_index = i * num_columns + j
            if image_index < len(book_images):
                cols[j].image(book_images[image_index], use_container_width=True)
                cols[j].write(book_names[image_index])
            else:
                cols[j].empty()  # If there are less than 4 movies in the last row, fill empty columns



def main():
    # Add custom CSS with glassmorphism and modern color palette
    st.markdown("""
        <style>
        /* Color palette */
        :root {
            --primary: #7F5AF0;
            --secondary: #2CB67D;
            --surface: rgba(255, 255, 255, 0.85);
            --background: #16161A;
            --headline: #FFFFFE;
            --paragraph: #94A1B2;
            --button: #7F5AF0;
            --button-text: #FFFFFE;
        }

        /* Main container styling */
        .main {
            background: linear-gradient(135deg, #2CB67D 0%, #7F5AF0 100%);
            padding: 2rem;
            min-height: 100vh;
        }

        /* Glassmorphism effect */
        .glass-effect {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }

        /* Sidebar styling */
        .css-1d391kg {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Headers styling */
        h1 {
            color: var(--headline);
            font-size: 2.5rem;
            text-align: center;
            margin-bottom: 2rem;
            padding: 1.5rem;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }

        h2 {
            color: var(--headline);
            margin-top: 2rem;
        }

        /* Card styling for bio sections */
        .stCard {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            margin: 1rem 0;
            transition: transform 0.3s ease;
        }

        .stCard:hover {
            transform: translateY(-5px);
        }

        /* Chat container styling */
        .chat-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 2rem;
            margin: 2rem auto;
            max-width: 800px;
        }

        /* Message bubbles */
        .message {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease;
        }

        .message:hover {
            transform: translateY(-2px);
        }

        /* Button styling */
        .stButton>button {
            background: var(--button) !important;
            color: var(--button-text) !important;
            border: none !important;
            padding: 0.6rem 1.2rem !important;
            border-radius: 10px !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
            backdrop-filter: blur(5px) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }

        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(127, 90, 240, 0.3) !important;
        }

        /* Input field styling */
        .stTextInput>div>div>input {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(5px) !important;
            border-radius: 10px !important;
            color: var(--headline) !important;
            padding: 0.8rem 1rem !important;
            transition: all 0.3s ease !important;
        }

        .stTextInput>div>div>input:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 2px rgba(127, 90, 240, 0.2) !important;
            background: rgba(255, 255, 255, 0.1) !important;
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            background: rgba(255, 255, 255, 0.05);
        }

        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(5px);
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(127, 90, 240, 0.5);
            border-radius: 4px;
            backdrop-filter: blur(5px);
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(127, 90, 240, 0.7);
        }

        /* Animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .animate-fade-in {
            animation: fadeIn 0.5s ease forwards;
        }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("🍿📚 Movie and Book Recommender 📚🍿")
    
    st.sidebar.write("🎉Welcome to our Recommender Chat🎉")
    
    st.sidebar.write("Navigate through the sections below:")
    page = st.sidebar.radio("🚀 Select a Section 🚀", ["🏠 Home", "🎬 Movies", "📚 Books", "🤖 Chatbot"])

    if page == "🏠 Home":
        st.title("🌟MOVIE AND BOOK RECOMMENDER BY OMMA🌟")
       
        st.write("Explore the realms of recommendations and insights!")
        st.image("welcomehomepage.jpg", use_container_width=True)
        
        # Meet the creators
        st.header("👩‍💻 About Us")
        st.markdown("---")

        # Create three columns for the team members
        col1, col2, col3 = st.columns(3)
        
        # Latif's bio
        with col1:
            st.subheader("🌟 Latif Abderrahmane")
            st.image("latif.jpg", use_container_width=True)  
            st.write("👨‍💼 La")  
            st.markdown("---")
           
        # Omar's bio
        with col2:
            st.subheader("🌟 Lgarch Omar")
            st.image("omar.jpg", use_container_width=True)  
            st.write("👨‍💼 Om")  
            st.markdown("---")
            
        # Soukrati's bio
        with col3:
            st.subheader("🌟 Mouslim Soukrati")
            st.image("soukrati.jpg", use_container_width=True)  
            st.write("👨‍💼 Mo")  
            st.markdown("---")
        st.header("🎯 Our Mission 🎯")
        st.markdown("---")
        st.write("Our mission is to revolutionize personalized recommendations and insights using cutting-edge data science and machine learning techniques. Our project is driven by a relentless pursuit of innovation and excellence. We aim to empower users by providing them with tailored movie and book recommendations, leveraging sophisticated algorithms and rich datasets. Our recommender system analyzes user preferences, historical ratings, and content features to deliver accurate and relevant suggestions. Through seamless integration of data processing, natural language processing, and deep learning, we strive to enhance the user experience and facilitate informed decision-making. With a commitment to continuous improvement and user satisfaction, we endeavor to craft a platform that enriches lives and inspires exploration.")

        # Footer
        st.write("---")
        st.write("Created by Lgarch & Latif & Soukrati")  

    elif page == "🤖 Chatbot":
        st.title("💬 Chat with the Recommender Bot 💬")
        st.write("Get personalized recommendations and insights...")
        chat()

    elif page == "🎬 Movies":
        display_movie_page()

    elif page == "📚 Books":
        display_book_page()

if __name__ == "__main__":
    main()