import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# instantiate empty lists
title = []
runtime = []
genre = []
rating = []
metascore = []
votes = []
gross = []
year = []
director = []
description = []
url_li = []

# create dictionary with all the list
movies = {
    'title': title,
    'runtime': runtime,
    'genre': genre,
    'rating': rating,
    'metascore': metascore,
    'votes': votes,
    'gross': gross,
    'year': year,
    'director': director,
    'description': description,
    'url': url_li
    }


def content_cleaner(soup):
    # from soup get all elements with class lister-item-content
    movie_containers = soup.find_all('div', class_ = 'lister-item-content')

    for movie in movie_containers:
        # from first movie_container get the title
        title.append(movie.h3.a.text)
        # from first movie_container get the runtime
        runtime.append(movie.find('span', class_ = 'runtime').text.split(' ')[0])
        # from first movie_container get the genre
        genre.append(movie.find('span', class_ = 'genre').text.strip())
        # from first movie_container get the rating
        rating.append(movie.strong.text)
        # boolean check for metascore
        if movie.find('span', class_ = 'metascore') is not None:
            # from first movie_container get the metascore
            metascore.append(movie.find('span', class_ = 'metascore').text)
        else:
            # from first movie_container get the metascore
            metascore.append(' ')
        # from first movie_container get the number of votes
        votes.append(movie.find('span', attrs = {'name':'nv'})['data-value'])
        # from first movie_container get the year
        year.append(movie.h3.find('span', class_ = 'lister-item-year text-muted unbold').text.strip('()'))
        # from first movie_container get the description
        description.append(movie.find_all('p', class_ = 'text-muted')[1].text.strip())
        # from first movie_container get the director
        director.append(movie.find_all('p', class_ = '')[0].a.text)
        # from first movie_container get the gross
        gross.append(movie.find('p', class_ ='sort-num_votes-visible').find_all('span', attrs = {'name':'nv'})[1].attrs['data-value'])
        # from movie_container get the url
        url_li.append('https://www.imdb.com/'+movie.h3.a['href'])

url = 'https://www.imdb.com/search/title/?groups=top_250&sort=user_rating,desc'

params = {
    'start': 1,
    'ref_': 'adv_nxt'
}


for i in range(5):
    params['start'] = 1+i*50
    # make request to the url with the parameters and get the response in english
    response = requests.get(url, params=params, headers={'Accept-Language': 'en-US, en;q=0.5'})
    soup = BeautifulSoup(response.text, 'html.parser')
    content_cleaner(soup)

df = pd.DataFrame(movies)

# replace ' ' with 0 in metascore column
df['metascore'] = df['metascore'].replace(' ', 0)
# replace ',' with '' in gross column
df['gross'] = df['gross'].str.replace(',', '')
# convert the columns to the correct data type
df['runtime'] = df['runtime'].astype(int)
df['metascore'] = df['metascore'].astype(int)
df['votes'] = df['votes'].astype(int)
df['gross'] = df['gross'].astype(int)

# save the dataframe as csv
df.to_csv('IMDB_Top_250.csv', index=False)
