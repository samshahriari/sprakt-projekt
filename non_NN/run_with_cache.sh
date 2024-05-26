#! /bin/sh
python3 language_eng.py -d -sum -loadcache -t data/test.csv

# For other configurations see below

#python3 language_eng.py -d -sum -tfidf -loadcache -t data/test.csv

#python3 language_eng.py -d -cos -loadcache -t data/test.csv

#python3 language_eng.py -d -cos -tfidf -loadcache -t data/test.csv

#python3 language_eng.py -d -mincos -loadcache -t data/test.csv

#python3 language_eng.py -d -mincos -tfidf -loadcache -t data/test.csv