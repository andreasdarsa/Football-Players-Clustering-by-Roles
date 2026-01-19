import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import euclidean_distances

# Preprocess the data
def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Only keep midfilders ('MF' or 'MF, FW' or 'MF, DF')
    df = df[df['Pos'].str.contains('MF', na=False)]
    # Keep players with at least 900 minutes played
    df = df[df['Min'] >= 900]

    # Save the preprocessed data
    df.to_csv('clean_data/preprocessed_players_data.csv', index=False)

    return df

# Select relevant features and normalize per 90 minutes
def select_features(df):
    features = [
        'npxG', 'xAG',
        'PrgP', 'PrgC', 'PrgR',
        'KP', 'PPA',
        'TklW', 'Int', 'Recov'
    ]

    df_per_90 = pd.DataFrame()
    for feature in features:
        df_per_90[feature + '_per_90'] = df[feature] / df['90s'] # Comparing stats per 90 makes comparisons fairer

    return df_per_90

# Scale data and apply PCA
def scale_pca(df):
    df = df.fillna(0)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    return pca_data

# Find optimal number of clusters using the elbow method
def elbow_method(data, max_k=10):
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

# Find similar players based on PCA coordinates
def find_similar_players(player_name, df, pca_matrix, n=5):
    try:
        # Find player index
        idx = df[df['Player'] == player_name].index[0]
        
        # Calculate distances from all other players in PCA space
        distances = euclidean_distances([pca_matrix[idx]], pca_matrix).flatten()
        
        # Find the n plus 1 closest (because the player itself has distance 0)
        similar_indices = distances.argsort()[1:n+1]
        
        print(f"Players similar to {player_name}:")
        return df.iloc[similar_indices][['Player', 'Squad', 'Cluster']]
    except IndexError:
        return "Player not found in the dataset."

if __name__ == "__main__":
    # 1. Prepare data for clustering
    df = preprocess_data('raw_data/players_data_light-2024_2025.csv')
    df_features = select_features(df)
    pca_data = scale_pca(df_features)

    # 2. Determine optimal number of clusters
    wcss = elbow_method(pca_data, max_k=10)
    for i, cost in enumerate(wcss, start=1):
        print(f'K={i}, WCSS={cost}')

    # Plot the elbow graph
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method For Optimal K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('WCSS')
    plt.show()

    # 3. Run k-means with k=5
    k = 5
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(pca_data)

    # 4. Visualize clusters with Plotly
    df_preprocessed = pd.read_csv('clean_data/preprocessed_players_data.csv')
    df_preprocessed['Cluster'] = clusters
    
    plot_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])
    plot_df['Player'] = df_preprocessed['Player']
    plot_df['Cluster'] = df_preprocessed['Cluster']

    fig = px.scatter(
        plot_df, x='PC1', y='PC2', 
        color='Cluster', hover_name='Player',
        hover_data=['Cluster'],
        title=f'Football Player Roles Clustering (k={k})',
        labels={'PC1': 'Progression and Attack', 
                'PC2': 'Creation and Defense'},
        template='plotly_dark',
        color_discrete_sequence=px.colors.qualitative.Vivid
    )

    fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='White')))
    fig.update_layout(xaxis_title="Progression and Attack",
                      yaxis_title="Creation and Defense",
                      legend_title="Cluster")
    
    fig.write_html('graphs/player_roles_clustering.html')

    # 5. Analyze clusters
    analysis = df_preprocessed.groupby('Cluster')[['npxG', 'xAG', 'PrgP', 'PrgC', 'TklW', 'Int']].mean()
    print(analysis)
    analysis.to_csv('clean_data/cluster_analysis.csv')

    # 6. Get input from user to find similar players
    while True:
        player_name = input("Enter a player's name to find similar players (or type 'exit' to quit): ")
        if player_name.lower() == 'exit':
            break
        similar_players = find_similar_players(player_name, df_preprocessed, pca_data, n=5)
        print(similar_players)