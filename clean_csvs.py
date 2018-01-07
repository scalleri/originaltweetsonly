#!/usr/bin/env python
# encoding: utf-8

""" MAPPING: #http://api.rubyonrails.org/classes/ActiveSupport/TimeZone.html """

import os
import csv
import sys
import pandas as pd
import json
import numpy as np
import codecs
from pytz import country_timezones


def main():
	directory = sys.argv[1]
	mapping_list = sys.argv[2]
	outfile_name = sys.argv[3]
	timezone_country = create_timzone_dict()
	mapping_dict = timezone_mapping(mapping_list)
	country_code_count =  check_for_timezone(directory,timezone_country,mapping_dict)

	with codecs.open('results.csv','w','utf-8') as results:
		for k,v in sorted(country_code_count.items()):
			results.write(k)
			results.write(',')
			results.write(str(v['count']))
			results.write('\n')

	with codecs.open(outfile_name + '.json','w','utf-8') as outjson:
		json.dump(country_code_count,outjson,ensure_ascii = False)


def timezone_mapping(mapping_list):
	mapping_dict = {}

	with codecs.open(mapping_list,'r','utf-8') as mapping:
		for line in mapping:
			line = line.split(' => ')
			mapping_dict[line[0]] = line[1].rstrip('\n')
			
	print mapping_dict
	return mapping_dict

def create_timzone_dict():
	# https://stackoverflow.com/questions/13020690/get-country-code-for-timezone-using-pytz

	timezone_country = {}
	for countrycode in country_timezones:
		timezones = country_timezones[countrycode]
		for timezone in timezones:
			timezone_country[timezone] = countrycode

	for k,v in timezone_country.items():
		print k, v

	return timezone_country

def check_for_timezone(directory,timezone_country,mapping_dict):
	file_list = os.listdir(directory)
	file_list = sorted(file_list)
	country_code_count = {}


	i = 0
	for filename in file_list:
		with codecs.open(directory+filename) as f:
			reader = csv.reader(f)
			row0 = next(reader)
			row1 = next(reader)
			
			if len(row1[3]) != 0:
				current_tz = row1[3]
				try:
					timezone = mapping_dict[current_tz]
				except:
					if current_tz in timezone_country:
						timezone = current_tz
					else:
						print current_tz
				try:
					#print timezone_country[timezone]
					if timezone_country[timezone] in country_code_count:
						country_code_count[timezone_country[timezone]]['count'] += 1
						country_code_count[timezone_country[timezone]]['user'].append(filename)

					else:
						country_code_count[timezone_country[timezone]] = {'count':1, 'user':[filename] } 
				except:
					print timezone, 'not found'
					
			

	return country_code_count
			
			


if __name__ == '__main__':
	main()