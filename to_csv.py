import csv

reviews=[]
sentiment_ratings = []

with open("data/Compiled_Reviews.txt") as f:
   for line in f.readlines()[1:]:
        fields = line.rstrip().split('\t')
        reviews.append(fields[0])
        sentiment_ratings.append(fields[1])

# Remove empty review
reviews.pop(33402)
sentiment_ratings.pop(33402)

rows = zip(reviews, sentiment_ratings)
with open('data/reviews.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['text', 'sentiment'])
    writer.writerows(rows)