#!/usr/bin/env python

import argparse
import os
from datetime import datetime
from pathlib import Path
from collections import Counter
from mpi4py import MPI
from dateutil.parser import parse
import numpy as np
import json
from collections import Counter
from collections import defaultdict
test_file= "testDataFiles/twitter-1mb.json"
large_test_file="testDataFiles/twitter-50mb.json"


'''
Measure execution time of each process by stopping the time via datetime.now()
and calculating the difference
'''

# Measure execution time from start of process to finish
START_TIME = datetime.now()
END_TIME = None

# Initialize MPI communication
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Count # of hashtags and languages in Twitter dataset')
parser.add_argument('--dataset', type=str, default='bigTwitter.json',
                    help='Path to Twitter dataset file')

print("TweeterData running on rank " + str(RANK) + " out of " + str(
    COMM.size) + " cores")

path_dataset_file = test_file
sentiment_hr_dict = []
max_tweets_by_day = []
max_tweets_count = None
max_tweets_day = None
tweets_by_day=defaultdict(int)
tweets_by_day_hour=defaultdict(int)
start_hour = 0
end_hour = 1

def getJsonData():
     with open(path_dataset_file, encoding="utf-8") as twitter_file:
                   tweet = json.load(twitter_file)
                   total_tweets=tweet["rows"]            
                   return total_tweets

def getDataToAnalyze(data):  
            
            for row in data:
                       try:
                            created_at = row["doc"]["data"]["created_at"]
                            day = created_at.split('T')[0]  # Extracting only the date                           
                            tweets_by_day[day] += 1
                            hour = int(created_at.split('T')[1].split(':')[0])  # Extract the hour component
                            day_hour_pair = (day, hour)
                            tweets_by_day_hour[day_hour_pair] += 1
                            timeTemp = parse(created_at)
                            hour= timeTemp.strftime('%Y-%m-%dT%H:%M:%SZ')
                            sentiment_hr_data ={}
                            sentiment_score = row["doc"]["data"]["sentiment"]                    
                            if sentiment_score is not None:
                                if isinstance(sentiment_score, dict):
                                    sentiment_score = sentiment_score["score"]                   
                                sentiment_score = float(sentiment_score)  
                                sentiment_hr_data[sentiment_score] = hour
                                sentiment_hr_dict.append(sentiment_hr_data)    
                       except KeyError:
                            continue
            return sentiment_hr_dict
    
        
def main():
    """Main method with implementation of scatter / gather logic
    Rank 0 (the master process) splits the dataset into smaller chunks
    """
    #take total tweet count
    total_tweets= getJsonData()        
    #divide and scatter chunks
    if RANK == 0:
        # Determine the size of each chunk
        chunk_size = len(total_tweets) // SIZE
        remainder = len(total_tweets) % SIZE  
        print(chunk_size)
        # Scatter the data
        data_to_scatter = [total_tweets[i * chunk_size:(i + 1) * chunk_size] for i in range(SIZE)]        
        # If there's a remainder, distribute it among the first few processes
        for i in range(remainder):
            data_to_scatter[i].append(total_tweets[chunk_size * SIZE + i]) 
    else:
        data_to_scatter = None       

   #gethappies hour    
    if SIZE < 2 and RANK == 0: 
        # Find the sentiment with the highest score
        max_sentiment_hr_dict= getDataToAnalyze(total_tweets)
        result = max(sentiment_hr_dict, key=lambda d: max(d.keys()))
        # Find the day with the most number of tweets
        max_tweets_day = max(tweets_by_day, key=tweets_by_day.get)
        max_tweets_count = tweets_by_day[max_tweets_day]
        print(result)
        print("Day with the maximum tweets:", max_tweets_day)
        print("Number of tweets on that day:", max_tweets_count) 
        # Find the hour with the most number of tweets
        
        tweets_by_day_hour_dict=[tweets_by_day_hour]
        print(tweets_by_day_hour_dict )
        total_tweets_in_range = 0
        for d in tweets_by_day_hour:
                start_hour = 0
                end_hour = 1
                for day_hour_pair, tweet_count in d.items():
                    day, hour = day_hour_pair
                    if hour >= start_hour and hour < end_hour:
                        total_tweets_in_range += tweet_count
                        start_hour += 1
                        end_hour += 1
        print("Total tweets between", start_hour, "and", end_hour, "hours on the same day:", total_tweets_in_range)
        # hour_with_most_tweets = max(tweets_by_day_hour, key=tweets_by_day_hour.get)
        # most_tweets_count = tweets_by_day_hour[hour_with_most_tweets]
        # print("hr with the maximum tweets:", hour_with_most_tweets)
        # print(" day:", most_tweets_count) 
    else:
        # Scatter the data
        chunk = COMM.scatter(data_to_scatter, root=0) 
        max_sentiment_hr_dict= getDataToAnalyze(chunk)
        max_sentiment_hr_dict = max(max_sentiment_hr_dict, key=lambda d: max(d.keys()))       
        
    #     # Print execution time for worker nodes
    # if RANK != 0:
    #     END_TIME = datetime.now()
    #     print("Execution time on core with rank " + str(RANK) + " was: " + str(
    #         END_TIME - START_TIME))

    if SIZE >2 :
    # Gather results in master process (rank 0) for sentiment hr dict
        worker_results = COMM.gather(max_sentiment_hr_dict, root=0)   
        if RANK ==0:
            result=  max(worker_results, key=lambda d: max(d.keys()))
            print(result)
           
    if SIZE >2 :
    # Gather results in master process (rank 0) for max of tweets per day
        worker_results2 = COMM.gather(tweets_by_day, root=0) 
        if RANK ==0:
            max_tweets_by_day = [(date, value) for d in worker_results2 for date, value in d.items()]
            print(max_tweets_by_day)
            sum_by_date = defaultdict(int)
            # Sum values for each date
            for date, value in max_tweets_by_day:
                sum_by_date[date] += value                                 
            max_tweets_day = max(sum_by_date, key=lambda x:x[1])
            max_tweets_count = sum_by_date[max_tweets_day]
            print("Day with the maximum tweets:", max_tweets_day)
            print("Number of tweets on that day:", max_tweets_count)

    if SIZE >2 :
    # Gather results in master process (rank 0) for max no of tweets per hr 
        worker_results3 = COMM.gather(tweets_by_day_hour, root=0) 
        print(worker_results3)
        if RANK ==0:
        # Hour range to count tweets
        # Iterate over data and count tweets within the hour range for the same day
            total_tweets_in_range = 0
            for d in worker_results3:
                start_hour = 0
                end_hour = 1
                for day_hour_pair, tweet_count in d.items():
                    day, hour = day_hour_pair
                    if hour >= start_hour and hour < end_hour:
                        total_tweets_in_range += tweet_count
                        start_hour += 1
                        end_hour += 1
        
        print("Total tweets between", start_hour, "and", end_hour, "hours on the same day:", total_tweets_in_range)

  # Print final results
    print("")
    print("Final results")
    END_TIME = datetime.now()
    print("Total execution time was: " + str(END_TIME - START_TIME))

  # Finalize MPI
    MPI.Finalize()  
    
if __name__ == "__main__":
    main() 