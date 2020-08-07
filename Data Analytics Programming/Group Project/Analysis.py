import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm


pd.set_option('display.max_columns', None)
pd.options.display.width = 0


df_spotify = pd.read_csv('Spotify-2000.csv')


print(df_spotify.sort_values(by='Popularity', ascending=False))

# print("', '".join(df_spotify.columns))



def kmeans(df):
    model = KMeans(init='k-means++', n_clusters=4, random_state=1)
    X = df[['Beats Per Minute (BPM)', 'Energy', 'Danceability', 'Loudness (dB)', 'Liveness', 'Valence',
            'Length (Duration)', 'Acousticness', 'Speechiness', 'Popularity']]
    X = X.replace(',', '', regex=True)
    X = X.astype(dtype='float')

    print(X.columns)
    scaler = MinMaxScaler()

    X = scaler.fit_transform(X)
    # X = X.values
    model.fit(X)
    a = 1
    b = 9
    plt.scatter(X[:, a], X[:, b], c=model.labels_, cmap='rainbow')
    plt.scatter(model.cluster_centers_[:, a], model.cluster_centers_[:, b], marker='x', s=200, linewidths=5)
    plt.show()


    # sum_of_distances = {}
    # for clusters in range(1, 30):
    #     kmeans = KMeans(n_clusters=clusters, random_state=1)
    #     kmeans.fit(X)
    #     sum_of_distances[clusters] = kmeans.inertia_
    # pd.Series(sum_of_distances).sort_index().plot()
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Sum of distances of points to closest cluster centers')
    # plt.show()


def linear(df, subset):
    """
    Attempting to predict popularity of a song given characteristics
    :param df:
    :return:
    """



    if subset:
        X = df[df['Top Genre'] == subset]
        X = X[['Year', 'Beats Per Minute (BPM)', 'Energy', 'Danceability', 'Loudness (dB)', 'Liveness',
               'Valence', 'Length (Duration)', 'Acousticness', 'Speechiness']]
        y = df[df['Top Genre'] == subset]['Popularity']

    else:
        X = df[['Year', 'Beats Per Minute (BPM)', 'Energy', 'Danceability', 'Loudness (dB)', 'Liveness',
                       'Valence', 'Length (Duration)', 'Acousticness', 'Speechiness']]
        y = df['Popularity']

    X = X.replace(',', '', regex=True)
    X = X.astype(dtype='float')



    model = sm.OLS(y, X)
    result = model.fit()
    print(result.summary())



# print(df_spotify['Top Genre'].value_counts().index)

# for genre in list(df_spotify['Top Genre'].value_counts().index):
#     linear(df_spotify, genre)



# kmeans(df_spotify)
# linear(df_spotify, False)