from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import firebase_admin
# from firebase_admin import credentials, firestore

app = Flask(__name__)

# cred = credentials.Certificate("C:/Users/Mohammed/ghazal-ab4a3-firebase-adminsdk-70ay9-32e5c060cb.json")
# firebase_admin.initialize_app(cred)
# db=firestore.client()


# Load datasets
data = pd.read_csv('copyy.csv', encoding='latin-1')
hotels_data = pd.read_csv('hotel_data.csv')

# Preprocess attractions data
selected_columns = ['Destination', 'Duration (days)', 'Activity Preference', 'Budget Range',
                    'Attraction_Name', 'Rating', 'Description', 'Duration (hours)',
                    'Admission_Price', 'Type', 'Destination.1', 'Photo_link', 'Latitude', 'Longitude']
selected_data = data[selected_columns].dropna(subset=['Admission_Price'])
text_columns = ['Activity Preference', 'Description', 'Type']
selected_data['combined_text'] = selected_data[text_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(selected_data['combined_text'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
def recommend_attractions(user_input):
    num_attractions_total = 2 * user_input['Duration (days)']
    attractions_per_day = 2
    num_days = user_input['Duration (days)']
    budget_range_mapping = {
        'Low': (0, 10),
        'Medium': (0, 25),
        'High': (0, 60)
    }
    selected_budget_range = budget_range_mapping.get(user_input['Budget Range'])
    selected_attractions = selected_data[
        (selected_data['Admission_Price'] >= selected_budget_range[0]) &
        (selected_data['Admission_Price'] <= selected_budget_range[1]) &
        (selected_data['Destination'] == user_input['Destination'])
    ]
    index_of_destination = selected_attractions[selected_attractions['Destination'] == user_input['Destination']].index[0]
    similarity_scores = []
    for idx, row in selected_attractions.iterrows():
        similarity_scores.append((idx, cosine_sim[index_of_destination][idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_attractions = similarity_scores[:num_attractions_total]

    recommendations = []
    for day in range(num_days):
        daily_recommendations = []
        for i in range(attractions_per_day):
            attraction_info = selected_data.loc[recommended_attractions[day * attractions_per_day + i][0]]
            daily_recommendations.append({
                'Attraction_Name': attraction_info['Attraction_Name'],
                'Rating': attraction_info['Rating'],
                'Description': attraction_info['Description'],
                'Duration (hours)': attraction_info['Duration (hours)'],
                'Admission_Price': attraction_info['Admission_Price'],
                'Photo_link': attraction_info['Photo_link'],
                'Latitude': attraction_info['Latitude'],
                'Longitude': attraction_info['Longitude']})
        recommendations.extend(daily_recommendations)
    return recommendations

@app.route('/get_attractions', methods=['POST'])
def get_attractions():
    # Handling attraction retrieval (same as previously defined)
    # Implementation...

            user_input = request.json
            filtered_data = data[
                (data['Destination'] == user_input['Destination']) &
                (data['Duration (days)'] == user_input['Duration (days)']) &
                (data['Activity Preference'] == user_input['Activity Preference']) &
                (data['Budget Range'] == user_input['Budget Range'])
            ]

            if filtered_data.empty:
                return jsonify({"message": "No attractions found for the given input."}), 404

            attraction_days = {}
            for day in range(1, user_input['Duration (days)'] + 1):
                attraction_days[f'Day {day}'] = []

            attraction_data = filtered_data[[
                'Attraction_Name', 'Rating', 'Description', 'Duration (hours)', 'Admission_Price', 'Photo_link',
                'Latitude', 'Longitude'
            ]].to_dict(orient='records')

            num_attractions_per_day = len(attraction_data) // user_input['Duration (days)']
            attraction_index = 0
            for day in range(1, user_input['Duration (days)'] + 1):
                for _ in range(num_attractions_per_day):
                    if attraction_index < len(attraction_data):
                        attraction_days[f'Day {day}'].append(attraction_data[attraction_index])
                        attraction_index += 1

            return jsonify(attraction_days)

       
def get_recommendations(user_input):
    try:
        attractions = recommend_attractions(user_input)
        return attractions
    except Exception as e:
        return {"message": str(e)}
def filter_hotels_by_destination(data, destination):
    return data[data['Destination'] == destination]

def recommend_hotels(destination_data):
    return destination_data[['Hotel_ Name', 'Address', 'Rate', 'Image_link', 'Hotel_link']]

@app.route('/recommend_hotels', methods=['POST'])
def get_hotel_recommendations():
    data = request.json
    destination = data.get('destination')

    if destination:
        destination_data = filter_hotels_by_destination(hotels_data, destination)
        if not destination_data.empty:
            recommendations = recommend_hotels(destination_data)
            return jsonify(recommendations.to_dict('records'))
        else:
            return jsonify({"message": "No hotels found for the specified destination."}), 404
    else:
        return jsonify({"message": "Please provide a destination parameter."}), 400

if __name__ == '__main__':
    app.run(debug=True, host="192.168.1.9", port=5000)
