from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def getStemmedReview(review):
    review = review.lower()
    
    # tokenize
    tokens = tokenizer.tokenize(review)

    # removing the stopwords
    new_tokens = [token for token in tokens if token not in sw]
    
    # stemming
    stemmed_tokens = [ps.stem(token) for token in new_tokens]
    
    cleaned_review = ' '.join(stemmed_tokens)

    return cleaned_review


def getStemmedDocument(document):
    d = []
    for doc in document:
        d.append(getStemmedReview(doc))
    return d

def prepare_message(messages):
    d = getStemmedDocument(messages)
    # very very import, dont do fit_transform!
    return cv.transform(d)

df = pd.read_csv('./spam.csv', encoding='ISO-8859-1')
le = LabelEncoder()

data = df.to_numpy()

y = data[:, 0]
X = data[:, 1]

X.shape, y.shape

tokenizer = RegexpTokenizer('\w+')
sw = set(stopwords.words('english'))
ps = PorterStemmer()



stemmed_document = getStemmedDocument(X)

stemmed_document[:10]

cv = CountVectorizer()

vectorized_corpus = cv.fit_transform(stemmed_document)

X = vectorized_corpus.todense()

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.33, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

model.score(X_test, y_test)


messages = [
    """Get 40% OFF on 
All Products 
This Year's End!
Buy ANY product license before 31 December 2020 at a HUGE 40% Discount
2020 has been a tough year, but we are determined to welcome 2021 on a high note
That's why we're offering a 20+20 Special end of year discount on all products
So get ready for an amazing 2021 with huge savings on the products you love
LEARN MORE 
Interested in renewing your license(s) instead? Get in touch with the Renewal Team
Online Version
""",
    
    
    """ 

Learn the best way to make a name for yourself online.
Sharing your site address on social media, business cards, or even in person needs to be quick and easy. Using a custom domain name is the single best way to appear more professional and make it easier for others to remember your site address. 
Now you can get a great domain name and forward it to kisanektamorcha.wordpress.com for as little as $15. 
Find your domain now!  


Start Simple, Build a Brand 
Forwarding your custom domain to your free site is the first step in building your brand and now you can do that for as little as $15. 
Every hour there are more than 12,000 new domains registered. Grab yours before someone else does! 
Choose From More than 300 Domain Extensions 
No matter what you do, or what you write about WordPress.com has a domain that will fit perfectly. 
You can choose from: 
•	Modern options including .blog, .design, .shop, .art, .page, .online, .link, .xyz, and hundreds of others. 
•	Popular country codes like .ca, .uk, .in, and more
•	And of course, classics like .com, .org, and .co
Find your domain now!  


Whatever the Future Holds, Your Domain is Flexible. 
Claiming your ideal domain name today allows you to get busy building a name for yourself. In the future the way you engage with your audience may change, but you’ll never have to worry about rebuilding an audience. 
WordPress.com will be ready to help you all along the journey. 


Download our free mobile app today.
View stats, moderate comments, create and edit posts, and upload media.
Click here to learn more. 

Automattic Inc. | 60 29th St. #343, San Francisco, CA 94110 
Unsubscribe from this email | Update your preferences 

""",

    """Hello hackers!

We'd like to congratulate you on your acceptance to BM VIII! We're thrilled that you'll be joining us for a weekend full of workshops, activities, and hacking.

Please check your application portal on our website (boilermake.org) for RSVPs and next steps. As always, feel free to email us at team@boilermake.org if you have any questions!

Best,
The BoilerMake Team .
""",

"""Hi HACK,
We’re excited to announce a big update: Gravit Designer’s new features are designed to help you create like a professional! 
Start your Free Trial of PRO to check out these features!
(just click on the 'Start Free Trial' on the bottom right of the app screen)
Here are just some of the amazing new features available only in Gravit Designer PRO:  
Touch support
•	Easily navigate the Gravit Designer PRO interface on Windows (including Surface and other touch-enabled devices), iPads, and Android tablets. 
•	Unleash your artistic potential with your fingers. Paint, draw, and interact with your vector designs like never before. 
•	To learn more about touch support, check out the following article: https://www.designer.io/en/gravit-designer/touch-design-feature 
Real-time collaboration
•	Work together on shared documents with annotations and comments. 
•	Assign specific roles when sharing documents. 
•	Invite collaborators without a Gravit Designer account to access your designs. 
•	To learn more about the new collaboration features, check out the following article: https://www.designer.io/en/gravit-designer/design-collaboration-feature 
Function history
•	Each action in your design Is tracked, allowing you to jump back in time easily.  
New Example files in the Welcome screen
•	Explore a rich – and growing – set of example designs to help you get started quickly.  
"""
]



messages = prepare_message(messages)

y_pred = model.predict(messages)

print(y_pred)