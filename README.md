# originaltweetsonly
The code repository for my bachelor thesis project: The Importance of Original Tweets: Content-Based Prediction of Twitter User Location

In order to run the code the Twitter API keys have to be put into the config.py file

Use the following command to install all the required dependencies:
``` 
$pip install -r requirements.txt 
$brew install coreutils #this installs gsplit which is needed later 
```


And run this command to create needed folders:

```$mkdir data models results user_lists usercsvs logfiles ```

The pipeline codes can be used independently but the whole pipeline looks like this:

``` $python3 twitter_stream_download.py -q [searchterm] -d data ```

Streams tweets with a searchterm and stores them one JSON Object per line in the folder 'data'.

``` $python3 parse_jsonfile.py data/stream_[searchterm.json] [id]```

Parses the JSON file of the tweets to extract users with specified metadata (here time_zone=True and lang='en') and stores a list of unique usernames under user_lists/user_list[id].txt.

``` 
$cd user_lists
$gsplit --additional-suffix .txt -d -l 40 user_list[id].txt
$rm user_list[id].txt
$cd .. 
``` 

Splits the userfile in batches of 40 users, to avoid timeouts when crawling the Twitter timelines.

``` $python3 batchslicer.py user_lists/ ```

Crawls the usertimelines of the specified users.

```
$cd usercsvs/
$find . -name ".csv" -size -50k -delete
$cd ..
```

Deletes empty user csv files

```$python3 clean_csv.py usercsvs/ mapping.txt [country:user_names.json]```

Extracts the user timezones and maps them to a list of countries. Creates a JSON with all the usernames per country.

```$python3 create_database.py usercsvs/ [user:tweets.json] -countries ["countries like: 'US,GB,IN'] [country:user_names.json] -eng [True/False] ```

Creates a database for the specified countries with the user and their tweets.

```$python3 classify.py [user:tweets.json] --users 500 [result file]```

Creates a multinomial sklearn model, stored under models of the n amount of users (here 500). 

