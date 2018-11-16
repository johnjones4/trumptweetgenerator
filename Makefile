DATADIR=$(shell pwd)/data

build:
	docker build . -t trumptweetgenerator
	mkdir $(DATADIR)

seed:
	docker run --volume "$(DATADIR):/data" trumptweetgenerator python seed_tweets.py

fetch:
	docker run --volume "$(DATADIR):/data" --env TWITTER_CONSUMER_KEY=$(TWITTER_CONSUMER_KEY) --env TWITTER_CONSUMER_SECRET=$(TWITTER_CONSUMER_SECRET) --env TWITTER_ACCESS_TOKEN_KEY=$(TWITTER_ACCESS_TOKEN_KEY) --env TWITTER_ACCESS_TOKEN_SECRET=$(TWITTER_ACCESS_TOKEN_SECRET) trumptweetgenerator python fetch_tweets.py

train:
	docker run --volume "$(DATADIR):/data" -d trumptweetgenerator python build_model.py

predict:
	docker run --volume "$(DATADIR):/data" -d trumptweetgenerator python make_predictions.py
