#!/usr/bin/python

import tweet_dumper
import os
import sys
import codecs
import datetime
from datetime import datetime

user_names_directory = sys.argv[1]
directory = os.listdir(user_names_directory)
directory = sorted(directory)

currentdate = datetime.now().strftime("%Y%m%d-%H%M%S")
logfile = codecs.open('logfiles/logfile'+currentdate+'.txt','w','utf-8')

for user_names_filename in directory:
	logfile.write(user_names_filename)
	logfile.write('\n')
	print '================ NEW FILE' + user_names_filename + "============="
	tweet_dumper.main(user_names_directory+user_names_filename)


	
	