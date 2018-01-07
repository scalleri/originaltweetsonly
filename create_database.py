#!/usr/bin/env/python3
# encoding: utf-8

""" this code takes the user csv files and creates a json file which will then be processed by the classifier"""

""" can cause error if \\: is included in the tweets """

import os
import csv
import sys
import codecs
import pandas as pd 
import json
import numpy
import codecs
import random 
import argparse
from polyglot.detect import Detector


parser = argparse.ArgumentParser(description='This code creates a database of a folder of csv files with user timelines')
parser.add_argument('usercsv_directory', action='store', help='the path to the directory with the user csv')
parser.add_argument('outfilename',action='store',help='name of the outfile')
parser.add_argument('-countries',action='store',type=str,help='pass the countries to select as "US,GB,IN,CA"')
parser.add_argument('jsondatabase',action='store',help='pass the jsondatabase created with clean csv')
parser.add_argument('-eng', help='only return english tweets')
parser.add_argument('-size',type=int,help='force db to be of certain size - default smallest count of user per country')

arguments = parser.parse_args()
directory = arguments.usercsv_directory
outfilename = arguments.outfilename
jsondatabase = arguments.jsondatabase
countries = arguments.countries
english = arguments.eng
size = arguments.size

if arguments.countries:
	countries = countries.split(',')
else:
	countries = ['US','GB','IN','CA']

if arguments.eng:
	english = True
else:
	english = False

if arguments.size:
	size = arguments.size
else:
	size = None



def main():
	user_list, data = select_users(jsondatabase,countries,size)
	table_dict = create_table(user_list,data)
	table_dict = language_detection(table_dict,english)
	write_db(table_dict)

#select user files from directory
def select_users(jsondatabase,countries,size):
	with codecs.open(jsondatabase,'r','utf-8') as db:
		data = json.load(db) 

	if size == None:
		size = min([len(data[country]['user']) for country in countries])
		
	
	print('size:', size)

	users = [ random.sample(data[country]['user'],size) for country in countries ]
	users = [x for userlist in users for x in userlist ]
	print('length users',len(users))

	return users,data


def write_db(table_dict):
	with codecs.open(outfilename+'.json','w','utf-8') as outfile:
		outfile.write(json.dumps(table_dict,ensure_ascii=False))

def language_detection(table_dict,english):
	if english == False:
		for k in list(table_dict):
			tweets = table_dict[k]['tweets'][0:7]
			tweets = (" ".join(tweets))
			

			try:
				detecor = Detector(tweets)
				conf = detecor.language.confidence
				if conf <= 70.0:
					table_dict['lang'] = 'en'
				else:
					table_dict['lang'] = 'non-eng'
			except:
				table_dict['lang'] = 'non-eng'
	else:
		for k in list(table_dict):
			tweets = table_dict[k]['tweets'][0:7]
			tweets = (" ".join(tweets))
			

			try:
				detecor = Detector(tweets)
				conf = detecor.language.confidence
				if conf <= 70.0:
					table_dict['lang'] = 'en'
			
			except:
				continue

	return table_dict
			

def create_table(user_files,data):
	table_dict = {}
	countryinfo = {user:k  for k,v in data.items() for user in v['user']}
	print(countryinfo)
	count = 0
	for file in user_files:
		userhandle = file.split('\n')[0]
		#print(file)
		with codecs.open(directory+'/'+file,'r',encoding='utf-8') as userfile:
			reader = csv.reader(userfile,delimiter=',')
			row0 = next(reader)
			row1 = next(reader)
			table_dict[userhandle]={'location':row1[2], 'timezone':row1[3], 'country': countryinfo[file], 'tweets':[row1[5]]}
			for line in reader:
				try: 
					#print("'" + line[5] + "'")
					table_dict[userhandle]['tweets'].append("'"+line[5]+"'")
				except IndexError:
					table_dict[userhandle]['tweets'].append("NaN")
		if count % 20:
			print("...",count)
	count += 1

	return table_dict



if __name__ == '__main__':
	main()