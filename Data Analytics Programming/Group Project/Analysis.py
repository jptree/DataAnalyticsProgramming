import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
import math


pd.set_option('display.max_columns', None)
pd.options.display.width = 0


df_spotify = pd.read_csv('Spotify-2000.csv')


# print(df_spotify['Top Genre'].value_counts())
# print("', '".join(df_spotify.columns))



def kmeans(df):
    model = KMeans(init='k-means++', n_clusters=10, random_state=1)
    df = df[df['Top Genre'] == 'album rock']
    # X = df[['Beats Per Minute (BPM)', 'Energy', 'Danceability', 'Loudness (dB)', 'Liveness', 'Valence',
    #         'Length (Duration)', 'Acousticness', 'Speechiness']]
    #

    X = df[['Beats Per Minute (BPM)', 'Acousticness', 'Speechiness']]

    # X = df[['Beats Per Minute (BPM)', 'Energy']]

    X = X.replace(',', '', regex=True)
    X = X.astype(dtype='float')

    print(X.columns)
    scaler = MinMaxScaler()

    XX = scaler.fit_transform(X)
    # X = X.values
    model.fit(XX)
    a = 0
    b = 1
    plt.scatter(XX[:, a], XX[:, b], c=model.labels_, cmap='rainbow')
    plt.scatter(model.cluster_centers_[:, a], model.cluster_centers_[:, b], marker='x', s=200, linewidths=5)
    plt.xlabel(str(X.columns[0]))
    plt.ylabel(str(X.columns[1]))
    plt.show()


    sum_of_distances = {}
    for clusters in range(1, 60, 5):
        kmeans = KMeans(n_clusters=clusters, random_state=1)
        kmeans.fit(X)
        sum_of_distances[clusters] = kmeans.inertia_
    pd.Series(sum_of_distances).sort_index().plot()
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of distances of points to closest cluster centers')
    plt.show()


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
    print(math.sqrt(result.mse_resid))


# Which genres were more popular coming through 1950s to 2000s?
def question_one(df):
    d = df.groupby(['Year'])['Top Genre'].value_counts()
    for i in range(1950, 2020):
        try:
            print(i, d.loc[i].axes[0].tolist()[0])
        except KeyError:
            print(i)


# Songs of which genre mostly saw themselves landing in the Top 2000s
def question_two(df):
    d = df['Top Genre'].value_counts()[:15]
    d.plot(kind='bar')
    plt.tight_layout()
    plt.show()
    # print(list(d.axes))

    # print(d[:10])


def summarize(df):
    return df.describe()


# print(summarize(df_spotify))
# kmeans(df_spotify)
# question_two(df_spotify)
# question_one(df_spotify)


def ridge_regression(data, predictors, alpha, models_to_plot={}):

    data = data.replace(',', '', regex=True)
    data = data.astype(dtype='float')

    ridgereg = Ridge(alpha=alpha, normalize=True)
    ridgereg.fit(data[predictors], data['Popularity'])
    y_pred = ridgereg.predict(data[predictors])


    # Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['Acousticness'], y_pred)
        plt.plot(data['Acousticness'], data['Popularity'], '.')
        plt.title('Plot for alpha: %.3g' % alpha)

    # Return the result in pre-defined format
    rrss = math.sqrt(sum((y_pred - data['Popularity']) ** 2))
    ret = [rrss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret


def lasso_regression(data, predictors, alpha, models_to_plot={}):
    data = data.replace(',', '', regex=True)
    data = data.astype(dtype='float')

    # Fit the model
    lassoreg = Lasso(alpha=alpha, normalize=True, max_iter=1e5)
    lassoreg.fit(data[predictors], data['Popularity'])
    y_pred = lassoreg.predict(data[predictors])

    # # Check if a plot is to be made for the entered alpha
    # if alpha in models_to_plot:
    #     plt.subplot(models_to_plot[alpha])
    #     plt.tight_layout()
    #     plt.plot(data['x'], y_pred)
    #     plt.plot(data['x'], data['y'], '.')
    #     plt.title('Plot for alpha: %.3g' % alpha)

    # Return the result in pre-defined format
    rrss = math.sqrt(sum((y_pred - data['Popularity']) ** 2))
    ret = [rrss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret





# Run Ridge Regression
models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
predictors = ['Year', 'Beats Per Minute (BPM)', 'Energy', 'Danceability', 'Loudness (dB)', 'Liveness', 'Valence',
            'Length (Duration)', 'Acousticness', 'Speechiness']
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
col = ['rrss', 'intercept'] + predictors
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

for i in range(10):
    coef_matrix_ridge.iloc[i,] = ridge_regression(df_spotify[['Year', 'Beats Per Minute (BPM)', 'Energy', 'Danceability', 'Loudness (dB)', 'Liveness', 'Valence',
            'Length (Duration)', 'Acousticness', 'Speechiness', 'Popularity']], predictors, alpha_ridge[i], models_to_plot)

print(coef_matrix_ridge)

coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)
for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(df_spotify[['Year', 'Beats Per Minute (BPM)', 'Energy', 'Danceability', 'Loudness (dB)', 'Liveness', 'Valence',
            'Length (Duration)', 'Acousticness', 'Speechiness', 'Popularity']], predictors, alpha_ridge[i], models_to_plot)

print(coef_matrix_lasso)

linear(df_spotify, False)