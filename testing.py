# Sneek peek on the results.
# Importing sentiment_mod module and then testing on some feeds and statements.

import sentiment_mod as senti

print(senti.sentiment("He is an incapable person. His projects are totally senseless."))
print(senti.sentiment("This movie was awesome! The acting was great, plot was wonderful !"))
print(senti.sentiment("This movie was utter junk.. I don't see what the point was at all. Horrible movie, 0/10"))
print(senti.sentiment("He is a freak."))
print(senti.sentiment("Movie was nice. Actors did very well. All together a nice experience."))
print(senti.sentiment("Are you fucking mad ?"))
print(senti.sentiment("You are dumb."))
