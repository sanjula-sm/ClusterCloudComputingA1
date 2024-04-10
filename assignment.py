#!/usr/bin/env python

import json
from collections import Counter
import argparse
import os
from datetime import datetime
from pathlib import Path
from collections import Counter
from mpi4py import MPI
import json
from collections import defaultdict
import time


#parse inpute file name
parser = argparse.ArgumentParser(description='Get commandline arguments.')
parser.add_argument("dataset", type=Path)
args = parser.parse_args()
path_dataset_file= args.dataset

# Initialize MPI communication
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Start timer
start_time = MPI.Wtime()

def createChunks(path_to_dataset, chunk_size, total_size):
    with open(path_to_dataset, 'rb') as f:
        chunk_end = f.tell()

        while True:
            chunk_start = chunk_end
            f.seek(f.tell() + chunk_size)
            f.readline()
            chunk_end = f.tell()
            if chunk_end > total_size:
                chunk_end = total_size
            yield chunk_start, chunk_end - chunk_start
            if chunk_end == total_size:
                break


def createBatch(path_to_dataset, chunk_start, chunk_size, BATCH_SIZE):
    with open(path_to_dataset, 'rb') as f:
        batch_end = chunk_start
        while True:
            batch_start = batch_end
            f.seek(batch_start + BATCH_SIZE)
            f.readline()
            batch_end = f.tell()
            if batch_end > chunk_start + chunk_size:
                batch_end = chunk_start + chunk_size
            yield batch_start, batch_end - batch_start
            if batch_end == chunk_start + chunk_size:
                break

def getPerDayMaxCount(max_by_days):
        sum_by_date=[]
        if max_by_days is not None:
            cleaned_data = [d[0] for d in max_by_days if d[0]]
            sum_counter = Counter()       
            for d in cleaned_data:
                    for key, value in d.items():
                        sum_counter[key] += value        
            sum_dict = dict(sum_counter)
            max_key = max(sum_dict, key=sum_dict.get)
            value = sum_dict[max_key]
            return max_key,value

def getPerHrPerDayMaxCount(max_by_hr_days):
      if max_by_hr_days is not None:            
            cleaned_data = [d[0] for d in max_by_hr_days if d[0]]
            sum_by_date_hr = {}
            for counts_dict in cleaned_data:
                    for key, value in counts_dict.items():
                        date, hour = key                    
                        sum_by_date_hr[(date, hour)] = sum_by_date_hr.get((date, hour), 0) + value
      max_date_hour = max(sum_by_date_hr, key=sum_by_date_hr.get)
      max_value = sum_by_date_hr[max_date_hour]
      return max_date_hour, max_value



class DataProcessor():

    def __init__(self, BATCH_SIZE):

        self.BATCH_SIZE = BATCH_SIZE
        self.tweets_by_days = defaultdict(int)
        self.tweet_count_per_day_hour = defaultdict(int)
        self.tweets_by_days_dict=[]
        self.tweet_count_per_day_hour_dict=[]
        self.sentiment_day_data=defaultdict(float)
        self.sentiment_day_dict=[]
        self.sentiment_day_hr_data=defaultdict(float)
        self.sentiment_day_hr_dict=[]

    def getResults(self):

        sum_dict=self.getMaxTweetPerDayCount()
        sum_hr_dict=self.getMaxTweetPerHrCount()
        sentiment_day_dict=self.getHappiestDay()
        sum_by_date_hr=self.getMaxSentPerHrCount()
        return sum_dict,sum_hr_dict,sentiment_day_dict,sum_by_date_hr

    def getMaxPerDayHrData(self,day_hr_dict):
        accumulated_counts=[]
        accumulated_counts = COMM.reduce(day_hr_dict, op=MPI.SUM, root=0)
        sum_by_date_hr = {}
        if accumulated_counts is not None:
            for counts_dict in accumulated_counts:
                for key, value in counts_dict.items():
                    date, hour = key                    
                    sum_by_date_hr[(date, hour)] = sum_by_date_hr.get((date, hour), 0) + value
        return sum_by_date_hr

    def getMaxPerDayData(self,day_dict):
        accumulated_counts=[]
        accumulated_counts = COMM.reduce(day_dict, op=MPI.SUM, root=0)
        sum_counter = Counter()       
        if accumulated_counts is not None:
            for d in accumulated_counts:
                for key, value in d.items():
                    sum_counter[key] += value        
        sum_dict = dict(sum_counter)
        return sum_dict
        
    def getMaxSentPerHrCount(self): 
        sum_by_date_hr=self.getMaxPerDayHrData(self.sentiment_day_hr_dict)
        return sum_by_date_hr

    def getHappiestDay(self):
        sum_dict=self.getMaxPerDayData(self.sentiment_day_dict)
        return sum_dict
        
    def getMaxTweetPerDayCount(self):
        sum_dict=self.getMaxPerDayData(self.tweets_by_days_dict)
        return sum_dict

    def getMaxTweetPerHrCount(self):
        sum_by_date_hr=self.getMaxPerDayHrData(self.tweet_count_per_day_hour_dict)
        return sum_by_date_hr
                  

    def process_tweet(self, tweet):
        try:
            created_at = tweet["doc"]["data"]["created_at"]                           
            tweet_time = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ")
            day_key = tweet_time.date()
            day_key = day_key.strftime("%Y-%m-%d") 
            hour_key = tweet_time.hour 
            self.tweets_by_days[day_key] += 1                       
            self.tweet_count_per_day_hour[(day_key, hour_key)] += 1
            sentiment_score = tweet["doc"]["data"]["sentiment"] 
            if isinstance(sentiment_score, dict):
                     sentiment_score = sentiment_score["score"]                   
                     sentiment_score = float(sentiment_score)  
            self.sentiment_day_data[day_key] += sentiment_score  
            self.sentiment_day_hr_data[(day_key, hour_key)] += sentiment_score
        except KeyError:
            pass
        
    def process_wrapper(self, path_to_dataset, chunk_start, chunk_size):

        with open(path_to_dataset, 'rb') as f:
            batches = []
            for read_start, read_size in createBatch(path_to_dataset, chunk_start,
                                                  chunk_size, self.BATCH_SIZE):
                batches.append({"batchStart": read_start, "batchSize": read_size})

            for batch in batches:
                f.seek(batch['batchStart'])

                if batch['batchSize'] > 0:
                    content = f.read(batch['batchSize']).splitlines()

                    for line in content:
                        line = line.decode('utf-8')              
                        if line[-1] == ",":
                            line = line[:-1]  
                        try:
                            tweet = json.loads(line)
                            self.process_tweet(tweet)                         
                        except Exception as e:
                           pass                      
                else:
                    print("batchsize with size 0 detected")
            self.tweets_by_days_dict.append(dict(self.tweets_by_days))
            self.tweet_count_per_day_hour_dict.append(dict(self.tweet_count_per_day_hour))
            self.sentiment_day_dict.append(dict(self.sentiment_day_data))
            self.sentiment_day_hr_dict.append(dict(self.sentiment_day_hr_data))
            
def main():
    
    max_tweets_by_days, max_tweet_by_day_hours,max_sent_by_days,max_sent_by_days_hr = [], [],[],[]
    batch_size= 1024 #default size
    data_processor = DataProcessor(batch_size)    
    if RANK == 0:
        dataset_size_total = os.path.getsize(path_dataset_file)
        dataset_size_per_process = dataset_size_total / SIZE
        chunks = []
        for chunkStart, chunkSize in createChunks(path_dataset_file,
                                              int(dataset_size_per_process),
                                              dataset_size_total):
            chunks.append({"chunkStart": chunkStart, "chunkSize": chunkSize})
    else:
        chunks = None
    COMM.Barrier()

    #scatter
    chunk_per_process = COMM.scatter(chunks, root=0)

    data_processor.process_wrapper(path_dataset_file,
                                   chunk_per_process["chunkStart"],
                                   chunk_per_process["chunkSize"])
    chunk_result = data_processor.getResults()  
    max_tweets_by_days.append(chunk_result[0])
    max_tweet_by_day_hours.append(chunk_result[1])
    max_sent_by_days.append(chunk_result[2])
    max_sent_by_days_hr.append(chunk_result[3])
    
    #gather
    max_tweets_by_days = COMM.gather(max_tweets_by_days, root=0) 
    max_tweet_by_day_hours = COMM.gather(max_tweet_by_day_hours, root=0)
    max_sent_by_days = COMM.gather(max_sent_by_days, root=0)
    max_sent_by_days_hr = COMM.gather(max_sent_by_days_hr, root=0)

    COMM.Barrier()

    #process and get final result
    if RANK == 0:

        max_date_hour, max_value=getPerHrPerDayMaxCount(max_sent_by_days_hr)
        hour_value= max_date_hour[1]+1 
        print(f"Happiest hour is {max_date_hour[1]} to {hour_value} on {max_date_hour[0]} with sentiment Score {max_value}")

        max_key,value=getPerDayMaxCount(max_sent_by_days)
        print(f"Happiest day is {max_key} with sentiment score {value} ")

        max_date_hour, max_value=getPerHrPerDayMaxCount(max_tweet_by_day_hours)
        hour_value= max_date_hour[1]+1 
        print(f"The Most active hour is {max_date_hour[1]} to {hour_value} on {max_date_hour[0]} with tweet count {max_value}")

        max_key,value=getPerDayMaxCount(max_tweets_by_days)
        print(f"The Most active day is {max_key} with tweet count {value} ")
        

    # Stop timer and get total elapsed time 
    end_time = MPI.Wtime()
    elapsed_time = end_time - start_time
    all_elapsed_times = COMM.gather(elapsed_time, root=0)
    if RANK == 0:
        total_elapsed_time = sum(all_elapsed_times)
        print(f"Total elapsed time:{total_elapsed_time} seconds")
        print(f"Total execution time: {elapsed_time} seconds")

 # Finalize MPI
    MPI.Finalize()  

if __name__ == "__main__":
    main()
