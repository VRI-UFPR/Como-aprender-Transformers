#!/bin/bash

wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
unzip -qq cornell_movie_dialogs_corpus.zip
mv "cornell movie-dialogs corpus" dataset
rm cornell_movie_dialogs_corpus.zip
