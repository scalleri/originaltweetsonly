#!/usr/bin/env python
# encoding: utf-8

""" This code gets all the (at least 3200) tweets of a user """

import tweepy #https://github.com/tweepy/tweepy
import csv
import sys
import os
import config

#Twitter API credentials
consumer_key = config.consumer_key
consumer_secret = config.consumer_secret
access_key = config.access_key
access_secret = config.access_secret

def main(user_list):
	user_names = open(user_list)
	user_list = user_names.readlines()
	for user in user_list:
		get_all_tweets(user)

def get_all_tweets(screen_name):
	#Twitter only allows access to a users most recent 3240 tweets with this method
	
	#authorize twitter, initialize tweepy
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_key, access_secret)
	api = tweepy.API(auth)
	
	#initialize a list to hold all the tweepy Tweets
	alltweets = []	
	
	#make initial request for most recent tweets (200 is the maximum allowed count)
	try:
		new_tweets = api.user_timeline(screen_name = screen_name,count=200)
	except tweepy.error.TweepError:
		return
	
	#save most recent tweets
	alltweets.extend(new_tweets)
	
	#save the id of the oldest tweet less one
	oldest = alltweets[-1].id - 1
	
	#keep grabbing tweets until there are no tweets left to grab
	while len(new_tweets) > 0:
		print("getting tweets before %s" % (oldest))
		try:
			#all subsiquent requests use the max_id param to prevent duplicates
			new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
		except tweepy.error.TweepError:
			return
		
		#save most recent tweets
		alltweets.extend(new_tweets)
		
		#update the id of the oldest tweet less one
		oldest = alltweets[-1].id - 1
		
		print("...%s tweets downloaded so far" % (len(alltweets)))
	
	#transform the tweepy tweets into a 2D array that will populate the csv	
	outtweets = [[tweet.id_str, tweet.created_at,tweet.user.location, tweet.user.time_zone, tweet.user.lang ,tweet.text.encode("utf-8")] for tweet in alltweets]
	
	#write the csv	
	with open('usercsvs/%s_tweets.csv' % screen_name, 'w') as f:
		writer = csv.writer(f)
		writer.writerow(["id","created_at","location","time_zone","lang","text",])
		try:
			writer.writerows(outtweets)
		except UnicodeEncodeError:
			return
	
	pass


if __name__ == '__main__':
	#pass in the username of the account you want to download
	main(user_list)
